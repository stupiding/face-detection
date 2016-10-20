#include <algorithm>
#include <cfloat>
#include <vector>

#include "caffe/layers/nms_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
void NMSLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  NMSParameter nms_param = this->layer_param_.nms_param();
  CHECK(!nms_param.has_kernel_size() !=
    !(nms_param.has_kernel_h() && nms_param.has_kernel_w()))
    << "Filter size is kernel_size OR kernel_h and kernel_w; not both";
  CHECK(nms_param.has_kernel_size() ||
    (nms_param.has_kernel_h() && nms_param.has_kernel_w()))
    << "For non-square filters both kernel_h and kernel_w are required.";
  if (nms_param.has_kernel_size()) {
    kernel_h_ = kernel_w_ = nms_param.kernel_size();
  } else {
    kernel_h_ = nms_param.kernel_h();
    kernel_w_ = nms_param.kernel_w();
  }
  
  CHECK_GT(kernel_h_, 0) << "Filter dimensions cannot be zero.";
  CHECK_GT(kernel_w_, 0) << "Filter dimensions cannot be zero.";
}

template <typename Dtype>
void NMSLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  CHECK_EQ(4, bottom[0]->num_axes()) << "Input must have 4 axes, "
      << "corresponding to (num, channels, height, width)";
  channels_ = bottom[0]->channels();
  height_ = bottom[0]->height();
  width_ = bottom[0]->width();
  top[0]->Reshape(bottom[0]->num(), channels_, height_, width_);
  if (top.size() > 1) {
    top[1]->ReshapeLike(*top[0]);
  }
}

template <typename Dtype>
void NMSLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  const Dtype* bottom_data = bottom[0]->cpu_data();
  Dtype* top_data = top[0]->mutable_cpu_data();
  Dtype* top_diff = top[0]->mutable_cpu_diff();

  const int top_count = top[0]->count();

  caffe_set(top_count, Dtype(0), top_diff);
  caffe_set(top_count, Dtype(0), top_data);

  // The main loop
  for (int n = 0; n < bottom[0]->num(); ++n) {
    for (int c = 0; c < channels_; ++c) {
      // NMS in column
      for (int h = 0; h < height_; ++h) {
        for (int w = 0; w < width_; ++w) {
          Dtype max_v = 0;
          for (int kh = 0; kh < kernel_h_; ++kh) {
            int nh = h + kh - kernel_h_ / 2;
            int nw = w;
            if(nh < 0 || nh >= height_) {
              continue;
            }
            if(max_v < bottom_data[nh * width_ + nw]) {
              max_v = bottom_data[nh * width_ + nw];
            }
          }
          if(max_v != bottom_data[h * width_ + w]) {
            top_diff[h * width_ + w] = max_v * Dtype(-1.);
          }
          else {
            top_diff[h * width_ + w] = max_v;
          }
        } // End of w loop
      } // End of h loop

      // NMS in row
      for (int h = 0; h < height_; ++h) {
        for (int w = 0; w < width_; ++w) {
          Dtype max_v = 0;
          for (int kw = 0; kw < kernel_w_; ++kw) {
            int nh = h;
            int nw = w + kw - kernel_w_ / 2;
            if(nw < 0 || nw >= width_) {
              continue;
            }
            if(max_v < abs(top_diff[nh * width_ + nw])) {
              max_v = abs(top_diff[nh * width_ + nw]);
            }
          }
          if(max_v == top_diff[h * width_ + w]) {
            top_data[h * width_ + w] = max_v; // * Dtype(-1.);
          }
        } // End of w loop
      } // End of h loop

      bottom_data += bottom[0]->offset(0, 1);
      top_diff += top[0]->offset(0, 1);
      top_data += top[0]->offset(0, 1);
    } // End of c loop
  } // End of n loop
}

template <typename Dtype>
void NMSLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  if (!propagate_down[0]) {
    return;
  }

  const Dtype* top_diff = top[0]->cpu_diff();
  const Dtype* top_data = top[0]->cpu_data();
  Dtype* bottom_diff = bottom[0]->mutable_cpu_diff();

  caffe_set(bottom[0]->count(), Dtype(0), bottom_diff);
  // The main loop
  for (int n = 0; n < top[0]->num(); ++n) {
    for (int c = 0; c < channels_; ++c) {
      for (int h = 0; h < height_; ++h) {
        for (int w = 0; w < width_; ++w) {
          const int index = h * width_ + w;
          if(top_data[index] != Dtype(0)) {
            bottom_diff[index] = top_diff[index];
          }
        } // End of w loop
      } // End of h loop

      bottom_diff += bottom[0]->offset(0, 1);
      top_diff += top[0]->offset(0, 1);
      top_data += top[0]->offset(0, 1);
    } // End of c loop
  }// End of n loop
}


#ifdef CPU_ONLY
STUB_GPU(NMSLayer);
#endif

INSTANTIATE_CLASS(NMSLayer);
REGISTER_LAYER_CLASS(NMS);

}  // namespace caffe
