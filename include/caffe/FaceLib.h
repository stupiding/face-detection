#ifndef FACELIB_H
#define FACELIB_H

#include <string>
#include <vector>
#include <numeric>
#include <iostream>
#include <memory>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include "caffe/blob.hpp"
#include "caffe/caffe.hpp"
#include "caffe/util/io.hpp"

using namespace cv;
using namespace std;
using namespace caffe;

static vector<int> DEFAULT_VECTOR;
static vector<vector<int> > DEFAULT_VV;

class FaceLib {
public:
  FaceLib();
  ~FaceLib();
  
  void initModel(const vector<string> prototxts, const vector<string> binarys, const int octave_level = 2);
  vector<pair<vector<int>, float> >  detect(const Mat image);
  vector<pair<vector<int>, float> > global_NMS(const vector<pair<vector<int>, float> > rects, float iou_thred = 0.8);
  
private:
  vector<pair<vector<int>, float> > predict(const Mat image, const vector<pair<vector<int>, float> > rects);
  void get_pyramid(const Mat image);
  void NMS(const float *src, float *dst, const int kernel_size, int height, int widht);

  void warpInputLayer(vector<Mat> *inputChannels, int level);
  void preprocess(Mat images, vector<Mat>* inputChannels, const vector<int> rect = DEFAULT_VECTOR);
  Mat cropImage(const Mat& im, const vector<int> rect);

public:
  int deviceID;
  int max_size_, min_size_;
  vector<float> thred_;

private: 
  vector<shared_ptr<Net<float> > >  nets_;
  vector<Mat> image_pyramid_;
  int nInputChannels_, net_stride_, octave_level_;
  vector<Size> inputSizes_;
};

#endif
