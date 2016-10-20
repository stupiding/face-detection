#include <algorithm>
#include <cfloat>
#include <vector>

#include "caffe/layers/nms_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
__global__ void NMSColForward(const int nthreads,
    const Dtype* const bottom_data, const int num, const int channels,
    const int height, const int width, const int kernel_h, const int kernel_w,
    Dtype* top_diff) {
  CUDA_KERNEL_LOOP(index, nthreads) {
    const int w = index % width;
    const int h = (index / width) % height;
    const int c = (index / width / height) % channels;
    const int n = index / width / height / channels;

    int hstart = h - kernel_h / 2;
    const int hend = min(hstart + kernel_h, height);
    hstart = max(hstart, 0);

    Dtype max_v = 0;
    const Dtype* const bottom_slice =
        bottom_data + (n * channels + c) * height * width;
    for (int kh = hstart; kh < hend; ++kh) {
      if (bottom_slice[kh * width + w] > max_v) {
          max_v = bottom_slice[kh * width + w];
      }
    }
    if(bottom_slice[h * width + w] == max_v) {
      top_diff[index] = max_v;
    } else {
      top_diff[index] = max_v * Dtype(-1.);
    }
  }
}

template <typename Dtype>
__global__ void NMSRowForward(const int nthreads, const int num, const int channels,
    const int height, const int width, const int kernel_h, const int kernel_w,
    Dtype* top_diff, Dtype* top_data) {
  CUDA_KERNEL_LOOP(index, nthreads) {
    const int w = index % width;
    const int h = (index / width) % height;
    const int c = (index / width / height) % channels;
    const int n = index / width / height / channels;

    int wstart = w - kernel_w / 2;
    const int wend = min(wstart + kernel_w, width);
    wstart = max(wstart, 0);

    Dtype max_v = 0;
    Dtype abs_b = 0;
    const Dtype* const top_slice =
        top_diff + (n * channels + c) * height * width;

    for (int kw = wstart; kw < wend; ++kw) {
      abs_b = top_slice[h * width + kw];
      abs_b = (abs_b > 0) ? abs_b : (-1 * abs_b);
      if (abs_b > max_v) {
          max_v = abs_b;
      }
    }

    if(top_slice[h * width + w] == max_v) {
      top_data[index] = max_v;
    } else {
      top_data[index] = Dtype(0.);
    }
  }
}
template <typename Dtype>
void NMSLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  const Dtype* bottom_data = bottom[0]->gpu_data();
  Dtype* top_data = top[0]->mutable_gpu_data();
  Dtype* top_diff = top[0]->mutable_gpu_data();
  int count = top[0]->count();

  NMSColForward<Dtype><<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(
      count, bottom_data, bottom[0]->num(), channels_,
      height_, width_, kernel_h_, kernel_w_, top_diff);

  NMSRowForward<Dtype><<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(
      count, bottom[0]->num(), channels_, height_, 
      width_, kernel_h_, kernel_w_, top_diff, top_data);

  CUDA_POST_KERNEL_CHECK;
}


template <typename Dtype>
__global__ void NMSBackward(const int nthreads, const Dtype* const top_diff,
    const Dtype* const top_data, Dtype* const bottom_diff) {
  CUDA_KERNEL_LOOP(index, nthreads) {
    bottom_diff[index] = (top_data[index] == Dtype(0.)) ? Dtype(0.) : top_diff[index];
  }
}

template <typename Dtype>
void NMSLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  if (!propagate_down[0]) {
    return;
  }
  const Dtype* top_diff = top[0]->gpu_diff();
  const Dtype* top_data = top[0]->gpu_data();
  Dtype* bottom_diff = bottom[0]->mutable_gpu_diff();
  const int count = bottom[0]->count();

  NMSBackward<Dtype><<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(
        count, top_diff, top_data, bottom_diff);
  CUDA_POST_KERNEL_CHECK;
}


INSTANTIATE_LAYER_GPU_FUNCS(NMSLayer);


}  // namespace caffe
