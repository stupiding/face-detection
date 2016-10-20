#ifndef CAFFE_DETECTION_DATA_LAYER_HPP_
#define CAFFE_DETECTION_DATA_LAYER_HPP_

#include <opencv2/core/core.hpp>

#include <string>
#include <utility>
#include <vector>
#include <map>

#include "caffe/blob.hpp"
#include "caffe/data_transformer.hpp"
#include "caffe/internal_thread.hpp"
#include "caffe/layer.hpp"
#include "caffe/layers/base_data_layer.hpp"
#include "caffe/proto/caffe.pb.h"

namespace caffe {

/**
 * @brief Provides data to the Net from image files.
 *
 * TODO(dox): thorough documentation for Forward and proto params.
 */
template <typename Dtype>
class DetectionDataLayer : public BasePrefetchingDataLayer<Dtype> {
 public:
  explicit DetectionDataLayer(const LayerParameter& param)
      : BasePrefetchingDataLayer<Dtype>(param) {}
  virtual ~DetectionDataLayer();
  virtual void DataLayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);

  virtual inline const char* type() const { return "DetectionData"; }
  virtual inline int ExactNumBottomBlobs() const { return 0; }
  virtual inline int ExactNumTopBlobs() const { return 2; }

 protected:
  shared_ptr<Caffe::RNG> prefetch_rng_;
  virtual void ShuffleImages();
  virtual void load_batch(Batch<Dtype>* batch);

  map<std::string, vector<std::string> > label_list_;
  map<std::string, cv::Mat> image_in_memory;
  vector<std::string> image_list_;
  int image_id_;
};


}  // namespace caffe

#endif  // CAFFE_DETECTION_DATA_LAYER_HPP_
