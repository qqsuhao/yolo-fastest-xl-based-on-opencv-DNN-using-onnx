#include "yolo.h"

YOLO::YOLO(Net_config config) {
	cout << "Net use " << config.netname << endl;
	this->confThreshold = config.confThreshold;				// 初始化置信度门限
	this->nmsThreshold = config.nmsThreshold;				// 初始化nms门限
	this->inpWidth = config.inpWidth;						// 初始化输入图像宽度
	this->inpHeight = config.inpHeight;						// 初始化输入图像高度
	strcpy_s(this->netname, config.netname.c_str());	// 初始化网络名称

	ifstream ifs(config.classesFile.c_str());
	string line;
	while (getline(ifs, line)) this->classes.push_back(line);		// 从coco.name加载类别名称

	this->net = readNetFromONNX(config.modelConfiguration);		// 加载网络文件和权重文件
	this->net.setPreferableBackend(DNN_BACKEND_OPENCV);		 // 根据计算机的配置设置加速方法，支持cpu，cuda，fpga，具体可以看源码
	this->net.setPreferableTarget(DNN_TARGET_CPU);
}


void YOLO::setcapSize(int width, int height) {
	this->capHeight = height;
	this->capWidth = width;
	this->scaleHeight = float(this->capHeight) / this->inpHeight;
	this->scaleWidth = float(this->capWidth) / this->inpWidth;
}


void YOLO::postprocess(Mat& frame, const vector<Mat>& outs)   // Remove the bounding boxes with low confidence using non-maxima suppression
{
	vector<int> classIds;
	vector<float> confidences;
	vector<float> scores;
	vector<Rect> boxes;

	for (size_t i = 0; i < outs.size(); ++i) {
		// Scan through all the bounding boxes output from the network and keep only the
		// ones with high confidence scores. Assign the box's class label as the class
		// with the highest score for the box.
		float* data = (float*)outs[i].data;
		for (int j = 0; j < outs[i].rows; ++j, data += outs[i].cols) {
			Mat score = outs[i].row(j).colRange(5, outs[i].cols);		// 使用80个类别的概率作为score
			Point classIdPoint;
			double max_score;
			// Get the value and location of the maximum score
			minMaxLoc(score, 0, &max_score, 0, &classIdPoint);		// 查询score中最大的元素及其位置
			if (data[4] > this->confThreshold) {		// data[4]是置信度
				int centerX = (int)(data[0] * this->scaleWidth);		// yolo的输出位置是相对于模型输入图像大小的
				int centerY = (int)(data[1] * this->scaleHeight);		// 因此需要进行简单的缩放
				int width = (int)(data[2] * this->scaleWidth);
				int height = (int)(data[3] * this->scaleHeight);
				int left = centerX - width / 2;
				int top = centerY - height / 2;

				classIds.push_back(classIdPoint.x);
				confidences.push_back(data[classIdPoint.x+5]);
				scores.push_back(max_score*data[4]);			
				boxes.push_back(Rect(left, top, width, height));
			}
		}
	}

	// Perform non maximum suppression to eliminate redundant overlapping boxes with
	// lower confidences
	vector<int> indices;
	NMSBoxes(boxes, scores, this->confThreshold, this->nmsThreshold, indices);  // 使用opencv自带的nms
	for (size_t i = 0; i < indices.size(); ++i) {
		int idx = indices[i];
		Rect box = boxes[idx];
		this->drawPred(classIds[idx], confidences[idx], box.x, box.y,
			box.x + box.width, box.y + box.height, frame);
	}
}

void YOLO::drawPred(int classId, float conf, int left, int top, int right, int bottom, Mat& frame)   // Draw the predicted bounding box
{
	//Draw a rectangle displaying the bounding box
	rectangle(frame, Point(left, top), Point(right, bottom), Scalar(0, 0, 255), 3);

	//Get the label for the class name and its confidence
	string label = format("%.2f", conf);
	if (!this->classes.empty()) {
		CV_Assert(classId < (int)this->classes.size());
		label = this->classes[classId] + ":" + label;
	}

	//Display the label at the top of the bounding box
	int baseLine;
	Size labelSize = getTextSize(label, FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseLine);
	top = max(top, labelSize.height);
	// rectangle(frame, Point(left, top - int(1.5 * labelSize.height)), Point(left + int(1.5 * labelSize.width), top + baseLine), Scalar(0, 255, 0), FILLED);
	putText(frame, label, Point(left, top), FONT_HERSHEY_SIMPLEX, 0.75, Scalar(0, 255, 0), 2);
}

void YOLO::detect(Mat& frame) {
	Mat blob;
	blobFromImage(frame, blob, double(1 / 255.0), Size(this->inpWidth, this->inpHeight), Scalar(0, 0, 0), true, false);
	// blobFromImage进行预处理：归一化，resize，减去均值，交换绿色和蓝色通道，不进行裁剪
	this->net.setInput(blob);		// 输入模型
	vector<Mat> outs_blob;
	vector<Mat> outs;
	vector<String> names = this->net.getUnconnectedOutLayersNames();		// 获取网络输出层信息
	this->net.forward(outs_blob, names);		// 模型的输出结果存入outs_blob，由于我们的yolo模型有两个yolo层，因此输出有两个
	int i = 0;
	for (i = 0; i < outs_blob.size(); i++) {
		vector<Mat> out;
		// 我们的onnx的输出是一个维度为 num_samples*1*(num_anchors*grid*grid)*6 的4维度矩阵，这是一个blob类型
		// 因此需要使用imagesFromBlob将blob转为mat；一个blob可能对应多个mat，个数即为num_samples
		// 由于我们只有一个样本，所以我们只有out[0]
		// 两个yolo层分别产生一个blob
		imagesFromBlob(outs_blob[i], out);			
		outs.push_back(out[0]);
	}
	this->postprocess(frame, outs);			// 后处理

	vector<double> layersTimes;
	double freq = getTickFrequency() / 1000;   // 用于返回CPU的频率。get Tick Frequency。这里的单位是秒
	double t = net.getPerfProfile(layersTimes) / freq;		// getPerfProfile 网络推理次数   次数/频率=时间
	string label = format("%s Inference time : %.2f ms", this->netname, t);
	putText(frame, label, Point(0, 30), FONT_HERSHEY_SIMPLEX, 1, Scalar(0, 0, 255), 2);
	//imwrite(format("%s_out.jpg", this->netname), frame);
}
