# yolo-fastest-xl based on opencv DNN using onnx

## Config

win10	VS2015	opencv4.5.1



## 介绍

- 基于opencv4.5.1的DNN模块实现yolo-fastest-xl的前向推理；
- 使用pytorch将模型导出为onnx格式，并使用DNN加载；
- 代码中加入较为详细的注释方便读者理解。



## 致谢参考代码

- [https://github.com/berak/opencv_smallfry/blob/605f5fdb4b55d8e5fe7e4c859cb0784d1007ffdd/demo/cpp/pnet.cpp](https://github.com/berak/opencv_smallfry/blob/605f5fdb4b55d8e5fe7e4c859cb0784d1007ffdd/demo/cpp/pnet.cpp)
- [https://github.com/hpc203/yolov34-cpp-opencv-dnn](https://github.com/hpc203/yolov34-cpp-opencv-dnn)