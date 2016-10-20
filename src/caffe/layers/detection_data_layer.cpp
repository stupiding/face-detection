#ifdef USE_OPENCV
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>

#include <fstream>  // NOLINT(readability/streams)
#include <iostream>  // NOLINT(readability/streams)
#include <string>
#include <utility>
#include <vector>
#include <map>

#include <time.h>
#include <stdlib.h>

#include "caffe/data_transformer.hpp"
#include "caffe/layers/base_data_layer.hpp"
#include "caffe/layers/detection_data_layer.hpp"
#include "caffe/util/benchmark.hpp"
#include "caffe/util/io.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/util/rng.hpp"

namespace caffe {

template <typename Dtype>
Dtype calculate_overlap(const vector<int> &rect1, const vector<int> &rect2) {
  int i_lx, i_ty, i_rx, i_by;
  i_lx = rect1[0] < rect2[0] ? rect2[0] : rect1[0];
  i_ty = rect1[1] < rect2[1] ? rect2[1] : rect1[1];
  i_rx = (rect1[0] + rect1[2]) < (rect2[0] + rect2[2]) ? (rect1[0] + rect1[2]) : (rect2[0] + rect2[2]);
  i_by = (rect1[1] + rect1[3]) < (rect2[1] + rect2[3]) ? (rect1[1] + rect1[3]) : (rect2[1] + rect2[3]);
  
  Dtype i_s = 0, sum_s = 0;
  if(i_lx < i_rx || i_ty < i_by) {
    i_s = (i_by - i_ty) * (i_rx - i_lx);
  }
  sum_s = (rect1[2] * rect1[3]) + (rect2[2] * rect2[3]) - i_s;

  return i_s / sum_s;
}

template <typename Dtype>
vector<int> get_pos_rect(vector<int> gt_rect, int img_w, int img_h, Dtype fg_thred) {
  vector<int> res_rect(4);
  Dtype iou;
  while(true) {
    int marg_x, marg_y;
    marg_x = int(gt_rect[2] * (1 - fg_thred));
    marg_y = int(gt_rect[3] * (1 - fg_thred));

    res_rect[0] = int(Dtype(rand()) / RAND_MAX * marg_x * 2 + gt_rect[0] - marg_x);
    res_rect[1] = int(Dtype(rand()) / RAND_MAX * marg_y * 2 + gt_rect[1] - marg_y);
    res_rect[2] = int(Dtype(rand()) / RAND_MAX * marg_x * 2 + gt_rect[2] - marg_x);
    res_rect[3] = int(Dtype(rand()) / RAND_MAX * marg_y * 2 + gt_rect[3] - marg_y);

    iou = calculate_overlap<Dtype>(gt_rect, res_rect);
    if(iou >= fg_thred) {
      res_rect[0] = res_rect[0] < 0 ? 0 : res_rect[0];
      res_rect[0] = res_rect[0] >= img_w ? img_w - 1 : res_rect[0];
      res_rect[1] = res_rect[1] < 0 ? 0 : res_rect[1];
      res_rect[1] = res_rect[1] >= img_h ? img_h - 1 : res_rect[1];

      res_rect[2] = (res_rect[2] + res_rect[0]) >= img_w ? (img_w - res_rect[0]) : res_rect[2];
      res_rect[3] = (res_rect[3] + res_rect[1]) >= img_h ? (img_h - res_rect[1]) : res_rect[3];

      return res_rect;
    }
  }
}

template <typename Dtype>
vector<int> get_neg_rect(vector<vector<int> > gt_rects, int img_w, int img_h, Dtype bg_thred) {
  vector<int> res_rect(4);
  Dtype iou, ciou;
  static int SMALLEST_SIDE = 3;
  while(true) {
    res_rect[0] = int(Dtype(rand()) / RAND_MAX * (img_w - SMALLEST_SIDE));
    res_rect[1] = int(Dtype(rand()) / RAND_MAX * (img_h - SMALLEST_SIDE));
    res_rect[2] = int(Dtype(rand()) / RAND_MAX * (img_w - res_rect[0] - SMALLEST_SIDE)) + SMALLEST_SIDE;
    res_rect[3] = int(Dtype(rand()) / RAND_MAX * (img_h - res_rect[1] - SMALLEST_SIDE)) + SMALLEST_SIDE;
    Dtype aspect_r = Dtype(res_rect[2]) / res_rect[3];
    if(aspect_r < 0.5 || aspect_r > 2) {
      continue;
    }

    iou = 0;
    for(size_t i = 0; i < gt_rects.size(); ++i) {
      ciou = calculate_overlap<Dtype>(gt_rects[i], res_rect);
      iou = ciou > iou ? ciou : iou;
    }
    if(iou < bg_thred) {
      return res_rect;
    }
  }
}

template <typename Dtype>
DetectionDataLayer<Dtype>::~DetectionDataLayer<Dtype>() {
  this->StopInternalThread();
}

template <typename Dtype>
void DetectionDataLayer<Dtype>::DataLayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  const int new_height = this->layer_param_.detection_data_param().new_height();
  const int new_width  = this->layer_param_.detection_data_param().new_width();
  const Dtype fg_thred = this->layer_param_.detection_data_param().fg_thred();
  const Dtype bg_thred = this->layer_param_.detection_data_param().bg_thred();
  const bool is_color  = this->layer_param_.detection_data_param().is_color();
  string root_folder = this->layer_param_.detection_data_param().root_folder();

  CHECK(new_height > 0 && new_width > 0) << "Current implementation requires "
      "new_height and new_width to be set at the same time.";
  CHECK((fg_thred < 1) && (bg_thred > 0) && (fg_thred >= bg_thred))
      << "The fg_thred or bg_thred is not set properly.";
  // Read the file with filenames and labels
  const string& source = this->layer_param_.detection_data_param().source();
  string image_name, label_str;
  LOG(INFO) << "Opening file " << source;
  std::ifstream infile(source.c_str());
  string line;
  size_t pos;
  while (std::getline(infile, line)) {
    pos = line.find('\t');
    image_name  = line.substr(0, pos);
    label_str = line.substr(pos + 1);
    if(std::find(image_list_.begin(), image_list_.end(), image_name) == image_list_.end()) {
      image_list_.push_back(image_name);
      vector<string> labels_str;
      labels_str.push_back(label_str);
      label_list_.insert(std::make_pair<string, vector<string> >(image_name, labels_str));
    }
    else {
      label_list_[image_name].push_back(label_str);
    }
  }

  CHECK(!image_list_.empty()) << "File is empty";

  if (this->layer_param_.detection_data_param().shuffle()) {
    // randomly shuffle data
    LOG(INFO) << "Shuffling data";
    const unsigned int prefetch_rng_seed = caffe_rng_rand();
    prefetch_rng_.reset(new Caffe::RNG(prefetch_rng_seed));
    ShuffleImages();
  }
  LOG(INFO) << "A total of " << image_list_.size() << " images.";

  image_id_ = 0;
  // Check if we would need to randomly skip a few data points
  if (this->layer_param_.detection_data_param().rand_skip()) {
    unsigned int skip = caffe_rng_rand() %
        this->layer_param_.detection_data_param().rand_skip();
    LOG(INFO) << "Skipping first " << skip << " data points.";
    CHECK_GT(image_list_.size(), skip) << "Not enough points to skip";
    image_id_ = skip;
  }

  vector<int> top_shape;
  top_shape.push_back(1);
  top_shape.push_back(is_color?3:1);
  top_shape.push_back(new_height);
  top_shape.push_back(new_width);

  this->transformed_data_.Reshape(top_shape);
  // Reshape prefetch_data and top[0] according to the batch_size.
  const int batch_size = this->layer_param_.detection_data_param().batch_size();
  const int pos_num = this->layer_param_.detection_data_param().pos_num();
  const int neg_num = this->layer_param_.detection_data_param().neg_num();
  CHECK_GT(batch_size, 0) << "Positive batch size required";
  CHECK(batch_size % (pos_num + neg_num) == 0)
    << "The batch size should be devisible by the sum of pos_num and neg_num.";
  top_shape[0] = batch_size;
  for (int i = 0; i < this->PREFETCH_COUNT; ++i) {
    this->prefetch_[i].data_.Reshape(top_shape);
  }
  top[0]->Reshape(top_shape);

  LOG(INFO) << "output data size: " << top[0]->num() << ","
      << top[0]->channels() << "," << top[0]->height() << ","
      << top[0]->width();

  // label
  vector<int> label_shape;
  label_shape.push_back(batch_size);
  label_shape.push_back(5);
  top[1]->Reshape(label_shape);
  for (int i = 0; i < this->PREFETCH_COUNT; ++i) {
    this->prefetch_[i].label_.Reshape(label_shape);
  }

  srand(time(0));
}

template <typename Dtype>
void DetectionDataLayer<Dtype>::ShuffleImages() {
  caffe::rng_t* prefetch_rng =
      static_cast<caffe::rng_t*>(prefetch_rng_->generator());
  shuffle(image_list_.begin(), image_list_.end(), prefetch_rng);
}

// This function is called on prefetch thread
template <typename Dtype>
void DetectionDataLayer<Dtype>::load_batch(Batch<Dtype>* batch) {
  CPUTimer batch_timer;
  batch_timer.Start();
  double read_time = 0;
  double trans_time = 0;
  CPUTimer timer;
  CHECK(batch->data_.count());
  CHECK(this->transformed_data_.count());
  DetectionDataParameter detection_data_param = this->layer_param_.detection_data_param();
  const int batch_size = detection_data_param.batch_size();
  const int pos_num = detection_data_param.pos_num();
  const int neg_num = detection_data_param.neg_num();
  const int img_num = batch_size / (pos_num + neg_num);
  const Dtype fg_thred = detection_data_param.fg_thred();
  const Dtype bg_thred = detection_data_param.bg_thred();
  const bool is_color = detection_data_param.is_color();
  const bool in_memory = detection_data_param.in_memory();
  string root_folder = detection_data_param.root_folder();

  Dtype* prefetch_data = batch->data_.mutable_cpu_data();
  Dtype* prefetch_label = batch->label_.mutable_cpu_data();

  // datum scales
  const int images_num = image_list_.size();
  for (int item_id = 0; item_id < img_num; ++item_id) {
    // get a blob
    timer.Start();
    CHECK_GT(images_num, image_id_);

    string image_name = image_list_[image_id_];
    cv::Mat cv_img;
    if(in_memory) {
      if(image_in_memory.find(image_name) != image_in_memory.end()) {
        cv_img = image_in_memory[image_name];
      }
      else {
        cv_img = ReadImageToCVMat(root_folder + image_name, 0, 0, is_color);
        image_in_memory.insert(std::make_pair<string, cv::Mat>(image_name, cv_img));
      }
    }
    else {
      cv_img = ReadImageToCVMat(root_folder + image_name, 0, 0, is_color);
    }
    CHECK(cv_img.data) << "Could not load " << image_name;
    read_time += timer.MicroSeconds();
    timer.Start();

    // Get the ground truth rects of the image
    vector<vector<int> > gt_rects;
    for(int gt_rect_id = 0; gt_rect_id < label_list_[image_name].size(); ++gt_rect_id) {
      vector<int> gt_rect;
      int x, y, w, h;
      sscanf(label_list_[image_name][gt_rect_id].c_str(), "%d %d %d %d", &x, &y, &w, &h);
      gt_rect.push_back(x);
      gt_rect.push_back(y);
      gt_rect.push_back(w);
      gt_rect.push_back(h);
      gt_rects.push_back(gt_rect);
    }

    // Apply transformations (mirror, crop...) to the image
    for(int rect_id = 0; rect_id < pos_num; ++rect_id) {
      int rects_num = gt_rects.size();
      int cur_gt_idx = rect_id % rects_num;
      int batch_idx = item_id * (pos_num + neg_num) + rect_id;
      int offset = batch->data_.offset(batch_idx);

      vector<int> t_rect = get_pos_rect(gt_rects[cur_gt_idx], cv_img.cols, cv_img.rows, fg_thred);
      cv::Rect rect(t_rect[0], t_rect[1], t_rect[2], t_rect[3]);

      CHECK(t_rect[2] * t_rect[3] > 0)
        << " Roi size if smaller than 0." << rect;

      cv::Mat resized_roi;
      cv::resize(cv_img(rect), resized_roi, cv::Size(batch->data_.width(), batch->data_.height()));

      this->transformed_data_.set_cpu_data(prefetch_data + offset);
      this->data_transformer_->Transform(resized_roi, &(this->transformed_data_));
      trans_time += timer.MicroSeconds();

      prefetch_label[batch_idx * 5] = 1;
      prefetch_label[batch_idx * 5 + 1] = float(t_rect[0] - gt_rects[cur_gt_idx][0]) / t_rect[2];
      prefetch_label[batch_idx * 5 + 2] = float(t_rect[1] - gt_rects[cur_gt_idx][1]) / t_rect[3];
      prefetch_label[batch_idx * 5 + 3] = float(t_rect[2] - gt_rects[cur_gt_idx][2]) / t_rect[2];
      prefetch_label[batch_idx * 5 + 4] = float(t_rect[3] - gt_rects[cur_gt_idx][3]) / t_rect[3];
    }

    for(int rect_id = 0; rect_id < neg_num; ++rect_id) {
      int batch_idx = item_id * (pos_num + neg_num) + pos_num + rect_id;
      int offset = batch->data_.offset(batch_idx);

      vector<int> t_rect = get_neg_rect(gt_rects, cv_img.cols, cv_img.rows,  bg_thred);
      cv::Rect rect(t_rect[0], t_rect[1], t_rect[2], t_rect[3]);

      CHECK(t_rect[2] * t_rect[3] > 0)
        << " Roi size if smaller than 0." << rect;

      cv::Mat resized_roi;
      cv::resize(cv_img(rect), resized_roi, cv::Size(batch->data_.width(), batch->data_.height()));


      this->transformed_data_.set_cpu_data(prefetch_data + offset);
      this->data_transformer_->Transform(resized_roi, &(this->transformed_data_));
      trans_time += timer.MicroSeconds();

      prefetch_label[batch_idx * 5] = 0;
      prefetch_label[batch_idx * 5 + 1] = 0;
      prefetch_label[batch_idx * 5 + 2] = 0;
      prefetch_label[batch_idx * 5 + 3] = 0;
      prefetch_label[batch_idx * 5 + 4] = 0;
    }


    // go to the next iter
    image_id_++;
    if (image_id_ >= images_num) {
      // We have reached the end. Restart from the first.
      DLOG(INFO) << "Restarting data prefetching from start.";
      image_id_ = 0;
      if (this->layer_param_.detection_data_param().shuffle()) {
        ShuffleImages();
      }
    }
  }
  batch_timer.Stop();
  DLOG(INFO) << "Prefetch batch: " << batch_timer.MilliSeconds() << " ms.";
  DLOG(INFO) << "     Read time: " << read_time / 1000 << " ms.";
  DLOG(INFO) << "Transform time: " << trans_time / 1000 << " ms.";
}

INSTANTIATE_CLASS(DetectionDataLayer);
REGISTER_LAYER_CLASS(DetectionData);

}  // namespace caffe
#endif  // USE_OPENCV
