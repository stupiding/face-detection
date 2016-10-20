#include <string>
#include <vector>
#include <numeric>
#include <iostream>
#include <memory>
#include <algorithm>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include "caffe/blob.hpp"
#include "caffe/caffe.hpp"
#include "caffe/util/io.hpp"
#include "caffe/proto/caffe.pb.h"

#include "caffe/FaceLib.h"

using namespace cv;
using namespace std;
using namespace caffe;

// set default mode CPU
//set max face size [0: the min side length of input image]
//set min face size [0: the input size of final network]
FaceLib::FaceLib():deviceID(-1),max_size_(0),min_size_(48) {} 
FaceLib::~FaceLib() {}

void get_params(string cfg_name, string param_name, string &param_value) {
  //ifstream in(cfg_name, ios::in);
}

static bool PairCompare_rects(const std::pair<vector<int>, float>& lhs,
                        const std::pair<vector<int>, float>& rhs) {
  return lhs.second > rhs.second;
}

void FaceLib::initModel(const vector<string> prototxts, const vector<string> binarys, const int octave_level){
    /* Set mode GPU/CPU */
  if (deviceID >= 0) {
    cerr << "Using GPU..." << endl;
    Caffe::set_mode(Caffe::GPU);
    Caffe::SetDevice(deviceID);
  } else
    Caffe::set_mode(Caffe::CPU);

  CHECK(prototxts.size() == binarys.size()) 
    << "The number of prototxts and binarys should be equal.";
  CHECK(octave_level > 0)
    << "The octave_level should be greater than 0.";

  /* Load the model */
  for(int i = 0; i < prototxts.size(); ++i) {
    shared_ptr<Net<float> > net;
    net.reset(new Net<float>(prototxts[i], TEST));
    net->CopyTrainedLayersFrom(binarys[i]);

    Blob<float>* inputLayer = net->input_blobs()[0];
    int output_num = net->num_outputs();
    nInputChannels_ = inputLayer->channels();

    CHECK(nInputChannels_ == 3)
      << "Input layer should have 1 or 3 channels.";
    if(i == prototxts.size() - 1)
      CHECK(output_num <= 3) << "In detection, only the number of output in the final net can be greater than 1.";
    else 
      CHECK(output_num == 1) << "In detection, only the number of output in the final net can be greater than 1.";

    if(output_num > 1) {
      Blob<float>* outputLayer1 = net->output_blobs()[0];
      Blob<float>* outputLayer2 = net->output_blobs()[1];
      CHECK(output_num <= 2) << "Not yet.";
      int out_channels1 = outputLayer1->channels();
      int out_channels2 = outputLayer2->channels();
      CHECK((out_channels1 <=2 && out_channels2 == 4) || (out_channels1 == 4 && out_channels2 <= 2)) 
          << "In detection, the number of channels in one of the outputlayers should be 1 or 2, and the other should be 4";
    } else {
      Blob<float>* outputLayer1 = net->output_blobs()[0];
      int pred_channels = outputLayer1->channels();
      CHECK(pred_channels <= 2) << "In detection, the number of channels in the prediction layer should be 1 or 2.";
    }

    Size inputSize = Size(inputLayer->height(), inputLayer->width());
    inputSizes_.push_back(inputSize);

    nets_.push_back(net);
  }
  thred_.push_back(0.3);
  thred_.push_back(0.3);
  thred_.push_back(0.3);

  Blob<float>* inputLayer = nets_[0]->input_blobs()[0];
  inputLayer->Reshape(1, nInputChannels_, inputSizes_[0].height * 2, inputSizes_[0].width * 2);
  nets_[0]->Reshape();
  Blob<float>* outputLayer = nets_[0]->output_blobs()[0];
  int height = outputLayer->height();
  net_stride_ = inputSizes_[0].height / (height - 1);

  octave_level_ = octave_level;
}

vector<pair<vector<int>, float> > FaceLib::detect(const Mat image) {
  //static int count = 0;
  Blob<float>* inputLayer = nets_[0]->input_blobs()[0];
  Size inputSize = inputSizes_[0];

  vector<pair<vector<int>, float> > detect_result, cascade_result;

  get_pyramid(image);

  if(image_pyramid_.size() == 0) {
    return cascade_result;
  }

  int pyramid_level = octave_level_ * (nets_.size() - 1);
  //int min_size = image.cols > image.rows ? image.rows : image.cols;
  //for(int l = 1; 1080 / 4 * pow(1.41421356, l) < min_size; l++, pyramid_level++) ;

  for(; pyramid_level < image_pyramid_.size(); ++pyramid_level) {

    Mat cur_image = image_pyramid_[pyramid_level];
    float resize_factor = float(cur_image.rows) / image.rows;

    int input_height = cur_image.rows > inputSize.height ? cur_image.rows : inputSize.height;
    int input_width = cur_image.cols > inputSize.width ? cur_image.cols : inputSize.width;    
    inputLayer->Reshape(1, nInputChannels_, input_height, input_width);    
    nets_[0]->Reshape();

    vector<Mat> inputChannels;
    warpInputLayer(&inputChannels, 0);
    preprocess(cur_image, &inputChannels);
    nets_[0]->Forward();

    Blob<float>* outputLayer = nets_[0]->output_blobs()[0];
    int height = outputLayer->height();
    int width = outputLayer->width();

    const float* output_data = outputLayer->cpu_data();
    
    if(outputLayer->channels() == 2) {
      output_data += outputLayer->offset(0,1);
    }

    float *nms_data;
    nms_data = (float*)malloc(height * width * sizeof(float));
    NMS(output_data, nms_data, 3, height, width);
    for(int r = 0; r < height; ++r) {
      for(int c = 0; c < width; ++c) {
        float rc_score = nms_data[r * width + c];
        if(rc_score > this->thred_[0]) {
        //if(output_data[r * width + c] > this->thred_) {
          vector<int> rect(5);
          rect[0] = int(this->net_stride_ * c / resize_factor);
          rect[1] = int(this->net_stride_ * r / resize_factor);
          rect[2] = int(inputSize.width / resize_factor);
          rect[3] = int(inputSize.height / resize_factor);
          rect[4] = pyramid_level;
          detect_result.push_back(make_pair<vector<int>, float>(rect, rc_score));
        }
      } // End of c loop
    } // End of r loop 
    free(nms_data);
  } // End of pyramid loop

  //cascade_result = detect_result; //predict(image, detect_result);
  cascade_result = predict(image, detect_result);
  //LOG(INFO) << count;
  return cascade_result;
}

vector<pair<vector<int>, float> > FaceLib::predict(Mat image, vector<pair<vector<int>, float> > rects) {

  vector<pair<vector<int>, float> > result_rects; 
  vector<int> is_face;

  for(int i = 0; i < rects.size(); ++i) {
    int pyramid_level = rects[i].first[4];
    for (int j = 1; j < nets_.size(); ++j) {
      vector<int> cur_rect(4);
      int cur_level = pyramid_level - j * octave_level_;
      float resize_factor = float(image_pyramid_[cur_level].rows) / image.rows;
      for(int k =0; k < 4; ++k) {
        cur_rect[k] = int(rects[i].first[k] * resize_factor);
      }
      cur_rect[2] = inputSizes_[j].height;
      cur_rect[3] = inputSizes_[j].width;
      vector<Mat> inputChannels;
      warpInputLayer(&inputChannels, j);

      preprocess(image_pyramid_[cur_level], &inputChannels, cur_rect);
      nets_[j]->Forward();

      Blob<float>* outputLayer, *rect_reg_layer = NULL;
      outputLayer = nets_[j]->output_blobs()[0];
      if(nets_[j]->num_outputs() == 2) {
        if(outputLayer->channels() == 4) {
          rect_reg_layer = outputLayer;
          outputLayer = nets_[j]->output_blobs()[1];
        } else {
          rect_reg_layer = nets_[j]->output_blobs()[1];
        }
      }
      const float* output_data = outputLayer->cpu_data();
      float face_score = output_data[0];
      if(outputLayer->channels() == 2) {
        face_score = output_data[1];
      }
      if(face_score < this->thred_[j]) {
        is_face.push_back(0);
        break;
      }
      else if (j == nets_.size() - 1) {
        is_face.push_back(1);
        if(rect_reg_layer == NULL) {
          result_rects.push_back(make_pair<vector<int>, float>(rects[i].first, face_score));
        } else {
          const float *rect_reg = rect_reg_layer->cpu_data();
          vector<int> new_rect(5);
          for(int k = 0; k < 4; k+=2) {
            new_rect[k] = rects[i].first[k] + rect_reg[k] * rects[i].first[2];
          }
          for(int k = 1; k < 4; k+=2) {
            new_rect[k] = rects[i].first[k] + rect_reg[k] * rects[i].first[3];
          }
          new_rect[4] = rects[i].first[4];
          result_rects.push_back(make_pair<vector<int>, float>(new_rect, face_score));
          
        }
      }
    } // End of nets_ loop
  } // End of rects loop 
  return result_rects;
}

vector<pair<vector<int>, float> > FaceLib::global_NMS(const vector<pair<vector<int>, float> > rects, float iou_thred) {
  vector<pair<vector<int>, float> > result_rects = rects;
  std::sort(result_rects.begin(), result_rects.end(), PairCompare_rects);
  for(int i = 0; i < result_rects.size(); ++i) {
    for(int j = result_rects.size() - 1; j > i; --j) {
      int i_lx, i_ty, i_rx, i_by;
      vector<int> rect1 = result_rects[i].first, rect2 = result_rects[j].first;
      i_lx = rect1[0] < rect2[0] ? rect2[0] : rect1[0];
      i_ty = rect1[1] < rect2[1] ? rect2[1] : rect1[1];
      i_rx = (rect1[0] + rect1[2]) < (rect2[0] + rect2[2]) ? (rect1[0] + rect1[2]) : (rect2[0] + rect2[2]);
      i_by = (rect1[1] + rect1[3]) < (rect2[1] + rect2[3]) ? (rect1[1] + rect1[3]) : (rect2[1] + rect2[3]);
  
      float i_s = 0;
      if(i_lx < i_rx || i_ty < i_by) {
        i_s = (i_by - i_ty) * (i_rx - i_lx);
      }
      if(i_s / (rect1[2] * rect1[3]) > iou_thred || i_s / (rect2[2] * rect2[3]) > iou_thred) {
        result_rects.erase(result_rects.begin() + j);
      }
    }
  }
  return result_rects;
}

void FaceLib::get_pyramid(Mat image) {
  for(int i = 0; i < image_pyramid_.size(); ++i) {
    image_pyramid_[i].release();
  }
  image_pyramid_.clear();

  float ratio_ = pow(2, (float(-1) / octave_level_));
  int net_num = nets_.size();
  int max_size = max_size_, min_size = min_size_; 
  int min_side = image.cols > image.rows ? image.rows : image.cols;
  int max_input = inputSizes_[net_num - 1].width, min_input = inputSizes_[0].width;

  if(max_size == 0 || max_size > min_side) {
    max_size = min_side;
  }
  if(max_size < min_size || max_size < max_input) {
    return;
  }

  float first_resize_factor = 1;
  if(min_size_ == 0 || min_size_ == max_input) {
    min_size = max_input;
    image_pyramid_.push_back(image);
  } else {
    Mat resized_image;
    first_resize_factor = float(max_input) / min_size;
    Size resized_size(int(image.cols * first_resize_factor), int(image.rows * first_resize_factor));

    resize(image, resized_image, resized_size);
    image_pyramid_.push_back(resized_image);
  }
  
  for(int pyramid_level = 1; 1; pyramid_level++) {
    float resize_factor = first_resize_factor * pow(ratio_, pyramid_level);
    Mat resized_image;
    Size resized_size(int(image.cols * resize_factor), int(image.rows * resize_factor));

    if(pyramid_level < octave_level_) {
      resize(image, resized_image, resized_size);
    } else {
      resize(image_pyramid_[pyramid_level - octave_level_], resized_image, resized_size);
    }
    image_pyramid_.push_back(resized_image);

    if(resize_factor * max_size < min_input) {
      break;
    }
  }
}

void FaceLib::NMS(const float *src, float *dst, const int kernel_size, int height, int width) {

  for(int i_d = 0; i_d < height; ++i_d) {
    for(int j_d = 0; j_d < width; ++j_d) {

      float max_value = src[i_d * width + j_d];
      int cur_ks = kernel_size; // (int)((3 - kernel_size / 2) * (1 - max_value * max_value)) * 2 + kernel_size;
      bool stop_flag = false;
      for(int di = 0; !stop_flag && di < cur_ks; ++di) {
        for(int dj = 0; !stop_flag && dj < cur_ks; ++dj) {
          int i_s = i_d + di - cur_ks / 2;
          int j_s = j_d + dj - cur_ks / 2;
          if(i_s < 0 || i_s >= height || j_s < 0 || j_s >= width) {
            continue;
          }
          if(max_value < src[i_s * width + j_s]) {
            max_value = 0;
            stop_flag = true;
          }
        } // End of dj loop
      } // End of di loop
      dst[i_d * width + j_d] = max_value;

    } // End of j_d loop
  } // End of i_d loop
}

void FaceLib::warpInputLayer(vector<Mat>* inputChannels, int level) {
    Blob<float>* inputLayer = nets_[level]->input_blobs()[0];
    int width = inputLayer->width();
    int height = inputLayer->height();
    float* data = inputLayer->mutable_cpu_data();
    for (int i = 0; i < inputLayer->channels(); ++i) {
        Mat channel(height, width, CV_32FC1, data);
        inputChannels->push_back(channel);
        data += width * height;
    }
    CHECK(reinterpret_cast<float*>(inputChannels->at(0).data)
            == nets_[level]->input_blobs()[0]->cpu_data())
        << "Input channels are not wrapping the input layer of the network.";
}

/* Convert the input image to the input image format of the network. */
void FaceLib::preprocess(Mat image, vector<Mat>* inputChannels, const vector<int> rect) {
  Mat im;
  Size inputSize = (*inputChannels)[0].size();

  if(rect.size() == 0) {
    if(image.cols < inputSizes_[0].width || image.rows < inputSizes_[0].height) {
      int bottom = image.rows < inputSizes_[0].height ? (inputSizes_[0].height - image.rows): 0;
      int right = image.cols < inputSizes_[0].width ? (inputSizes_[0].width - image.cols) : 0;
      //value = Scalar(0, 0, 0);
      copyMakeBorder(image, im, 0, bottom, 0, right, BORDER_CONSTANT);
    } else {
      im = image;
    }
  }
  else {
    Mat cropped_im = cropImage(image, rect);
    if (cropped_im.size() != inputSize) {
      CHECK(cropped_im.rows <= inputSize.height && cropped_im.cols <= inputSize.width)
          << "Error crop " << cropped_im.rows << ":" << cropped_im.cols;
      int bottom = cropped_im.rows < inputSize.height ? (inputSize.height - cropped_im.rows): 0;
      int right = cropped_im.cols < inputSize.width ? (inputSize.width - cropped_im.cols) : 0;
      //value = Scalar(0, 0, 0);
      copyMakeBorder(cropped_im, im, 0, bottom, 0, right, BORDER_CONSTANT);
      //resize(cropped_im, im, inputSize);
    } else
      im = cropped_im;
  }
  /* Convert to float */
  vector<Mat> channels;

  if(nInputChannels_ == 3) {
    im.convertTo(im, CV_32FC3);
    split(im, channels);
  } else {
    im.convertTo(im, CV_32FC1);
    channels.push_back(im);
  }

  for (int j = 0; j < channels.size(); j++) {
    channels[j].copyTo((*inputChannels)[j]);
  }
}

Mat FaceLib::cropImage(const Mat& im, const vector<int> rect) {

  int x = rect[0], y = rect[1], w = rect[2], h = rect[3];
  w = (x + w > im.cols) ? (im.cols - x) : w;
  h = (y + h > im.rows) ? (im.rows - y) : h;
  Rect new_rect(x, y, w, h);
  return im(new_rect);
}

float calculate_overlap(const vector<int> &rect1, const vector<int> &rect2) {
  int i_lx, i_ty, i_rx, i_by;
  i_lx = rect1[0] < rect2[0] ? rect2[0] : rect1[0];
  i_ty = rect1[1] < rect2[1] ? rect2[1] : rect1[1];
  i_rx = (rect1[0] + rect1[2]) < (rect2[0] + rect2[2]) ? (rect1[0] + rect1[2]) : (rect2[0] + rect2[2]);
  i_by = (rect1[1] + rect1[3]) < (rect2[1] + rect2[3]) ? (rect1[1] + rect1[3]) : (rect2[1] + rect2[3]);
  
  float i_s = 0, sum_s = 0;
  if(i_lx < i_rx || i_ty < i_by) {
    i_s = (i_by - i_ty) * (i_rx - i_lx);
  }
  sum_s = (rect1[2] * rect1[3]) + (rect2[2] * rect2[3]) - i_s;

  return i_s / sum_s;
}

int main() {
    // Load model
    const string prototxt1 = "./model1.prototxt";
    const string prototxt2 = "./model2.prototxt";
    const string prototxt3 = "./model3.prototxt";
    const string binary = "./model.caffemodel";

    vector<string> prototxts, binarys;
    prototxts.push_back(prototxt1);
    prototxts.push_back(prototxt2);
    prototxts.push_back(prototxt3);

    binarys.push_back(binary);
    binarys.push_back(binary);
    binarys.push_back(binary);

    FaceLib p;
    //p.deviceID = 0; // -1 for CPU, others for GPU
    p.initModel(prototxts, binarys, 2);

    // Load test image
    Mat images;
    std::ifstream infile("test.txt");
    string line, label_str;
    string im_name, im_path;

    size_t pos;
    int acc_num5 = 0, gt_a_num5 = 0, gt_num = 0, det_num = 0;
    int acc_num8 = 0, gt_a_num8 = 0;
    while (std::getline(infile, line)) {
      pos = line.find('\t');
      im_name  = line.substr(0, pos);
      label_str = line.substr(pos + 1);

      vector<int> gt_rect;
      int x, y, w, h;
      sscanf(label_str.c_str(), "%d %d %d %d", &x, &y, &w, &h);
      gt_rect.push_back(x);
      gt_rect.push_back(y);
      gt_rect.push_back(w);
      gt_rect.push_back(h);
      gt_num += 1;
      if(gt_num % 100 == 0) {
        LOG(INFO) << "Detected " << gt_num << " images.";
      }
      im_path = "/home/guojinma/Datasets/face/AFLW/data/data/flickr/" + im_name;
      Mat im = imread(im_path.c_str());
      if (im.empty()) {
        cerr << "Test image is empty:" << im_path << endl;
        return -1;
      }
      // Detect
      vector<pair<vector<int>, float> > locations_det, locations;
      locations_det = p.detect(im);
      locations = p.global_NMS(locations_det, 0.6);
      //locations = locations_det; //p.global_NMS(locations_det, 0.8);
      det_num += locations.size();
      bool Fir_flag5 = true;
      bool Fir_flag8 = true;
      for (size_t j = 0; j < locations.size(); ++j) {
        Rect new_rect(locations[j].first[0], locations[j].first[1], locations[j].first[2], locations[j].first[3]);
        float iou = calculate_overlap(locations[j].first, gt_rect);
        if(iou > 0.5) {
          acc_num5 += 1;
          if(Fir_flag5) {
            gt_a_num5 +=1;
            Fir_flag5 = false;
          }
          //rectangle(im, new_rect, Scalar(0,0,255), 3);
        }
        if(iou > 0.7) {
          acc_num8 += 1;
          if(Fir_flag8) {
            gt_a_num8 +=1;
            Fir_flag8 = false;
          }
        }
        char score_str[12];
        sprintf(score_str, "%.3f", locations[j].second);
        putText(im, score_str, Point(locations[j].first[0], locations[j].first[1]), FONT_HERSHEY_SCRIPT_SIMPLEX, 0.7, Scalar(0, 0, 255), 2, 8);
        rectangle(im, new_rect, Scalar(0,0,255), 3);
      }
      im_path = "./test_results/" + im_name;
      imwrite(im_path.c_str(), im);
  }

  LOG(INFO) << "0.5 Recall: " << float(gt_a_num5) / gt_num << "\t" << gt_a_num5 << "/" << gt_num;
  LOG(INFO) << "0.5 Prection: " << float(acc_num5) / det_num << "\t" << acc_num5 << "/" << det_num;

  LOG(INFO) << "0.7 Recall: " << float(gt_a_num8) / gt_num << "\t" << gt_a_num8 << "/" << gt_num;
  LOG(INFO) << "0.7 Prection: " << float(acc_num8) / det_num << "\t" << acc_num8 << "/" << det_num;
    return 0;
}
