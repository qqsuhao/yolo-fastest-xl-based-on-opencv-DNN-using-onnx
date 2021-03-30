#pragma once
#include <fstream>
#include <sstream>
#include <iostream>
#include <opencv2/dnn.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/core.hpp>
#include <opencv2/dnn/layer.details.hpp>

using namespace cv;
using namespace dnn;
using namespace std;

struct Net_config				// 网络配置
{
	float confThreshold;		// 置信度门限
	float nmsThreshold;			// NMS门限
	int inpWidth;				// 模型输入图像宽度
	int inpHeight;				// 模型输出图像高度
	string classesFile;			// 存放类别名称的文件 coco.name
	string modelConfiguration;	// 模型文件  .cfg
	string modelWeights;		// 模型权重文件
	string netname;				// 网络名称
};


class YOLO {
public:
	YOLO(Net_config config);				// 构造函数
	void detect(Mat& frame);				// 进行检测
	void setcapSize(int width, int height);		// 获取摄像头的分辨率
private:
	float confThreshold;			// 置信度门限
	float nmsThreshold;				// nms门限
	int inpWidth;					// 模型输入图像宽度
	int inpHeight;					// 模型输入图像高度
	int capWidth;					// 相机拍摄图像的宽度
	int capHeight;					// 相机拍摄图像的高度
	float scaleHeight;				// 模型输入图像到相机拍摄图像的高度缩放因子
	float scaleWidth;				// 模型输入图像到相机拍摄图像的宽度缩放因子
	char netname[20];				// 网络名字
	vector<string> classes;			// 存放类别的名字
	Net net;						// dnn::net类型
	void postprocess(Mat& frame, const vector<Mat>& outs);			// 后处理，主要是使用nms筛选目标
	void drawPred(int classId, float conf, int left, int top, int right, int bottom, Mat& frame);	// 绘制目标的检测结果以及置信度等参数
};


class ExpLayer : public cv::dnn::Layer {       // 自定义网络层Exp，参考https://github.com/berak/opencv_smallfry/blob/605f5fdb4b55d8e5fe7e4c859cb0784d1007ffdd/demo/cpp/pnet.cpp
public:
	ExpLayer(const cv::dnn::LayerParams &params) : Layer(params) {
	}

	static cv::Ptr<cv::dnn::Layer> create(cv::dnn::LayerParams& params) {
		return cv::Ptr<cv::dnn::Layer>(new ExpLayer(params));
	}

	virtual bool getMemoryShapes(const std::vector<std::vector<int> > &inputs,
		const int requiredOutputs,
		std::vector<std::vector<int> > &outputs,
		std::vector<std::vector<int> > &internals) const CV_OVERRIDE {
		CV_UNUSED(requiredOutputs); CV_UNUSED(internals);
		std::vector<int> outShape(4);
		outShape[0] = inputs[0][0];  // batch size
		outShape[1] = inputs[0][1];  // number of channels
		outShape[2] = inputs[0][2];
		outShape[3] = inputs[0][3];
		outputs.assign(1, outShape);
		return false;
	}

	virtual void forward(cv::InputArrayOfArrays inputs_arr,
		cv::OutputArrayOfArrays outputs_arr,
		cv::OutputArrayOfArrays internals_arr) CV_OVERRIDE {
		std::vector<cv::Mat> inputs, outputs;
		inputs_arr.getMatVector(inputs);
		outputs_arr.getMatVector(outputs);

		cv::Mat& inp = inputs[0];
		cv::Mat& out = outputs[0];

		exp(inp, out);			// 关键的一句代码
	}
};