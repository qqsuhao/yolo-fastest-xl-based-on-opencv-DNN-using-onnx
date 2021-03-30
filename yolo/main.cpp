#include "yolo.h"

using namespace cv;
using namespace std;


int main() {
	CV_DNN_REGISTER_LAYER_CLASS(Exp, ExpLayer);		// 添加自定义的网络层
	
	Net_config yolo_nets[1] = {
		{ 0.5f, 0.3f, 320, 320,"coco.names", "yolo-fastest-xl.onnx", "yolo-fastest-xl.onnx", "yolo-fastest-xl" }
	};

	YOLO yolo_model(yolo_nets[0]);

	static const string kWinName = "Deep learning object detection in OpenCV";
	namedWindow(kWinName, WINDOW_KEEPRATIO);
	VideoCapture cap;
	cap.open(0);
	Mat srcimg;
	while (1) {
		cap >> srcimg;
		yolo_model.setcapSize(srcimg.cols, srcimg.rows);
		yolo_model.detect(srcimg);
		imshow(kWinName, srcimg);
		waitKey(10);
	}
	destroyAllWindows();

	return 0;
}
