# MotionTrack-TensorRT in C++
MotionTrackCpp is the deploy version of MotionTrack using TensorRT in Cpp. MotionTrackCpp has been successfully tested in Nvidia RTX 3060 Laptop (x86-64) and Jetson Xavier NX (AArch64). Other similar platforms should also be able to use MotionTrackCpp.

# Requirement of Nvidia RTX 3060 Laptop
* eigen3.3.8
* opencv
* cuda11.4
* cudnn8.2.4
* tensorrt8.2.0.6 or 8.4.1.5

# Requirement of Jetson Xavier NX
* Jetpack5.0.2

# Installation
* Install opencv with ```sudo apt-get install libopencv-dev``` (we don't need a higher version of opencv like v3.3+).
* Download and unzip [eigen3.3.8](https://gitlab.com/libeigen/eigen/-/archive/3.3.8/eigen-3.3.8.zip). When using, you only need to include the header file in [CMakeLists.txt](CMakeLists.txt).
* Please follow the [TensorRT](https://docs.nvidia.com/deeplearning/tensorrt/install-guide/index.html) Installation Guide to install TensorRT. 

# Convert the onnx model to the TensorRT model
Place the onnx model in the [model](model) folder. For the generation of the onnx model, refer to our project [https://github.com/lzq11/MotionTrack](https://github.com/lzq11/MotionTrack). Please modify your own trtexec path.
## Nvidia RTX 3060 Laptop
```shell
# yolov7-w6
/home/lzq/Lib/TensorRT-8.4.1.5/bin/trtexec --onnx=model/W6-1920.onnx --saveEngine=model/W6-1920.trt --fp16 --workspace=5120
# yolov7-tiny 
/home/lzq/Lib/TensorRT-8.4.1.5/bin/trtexec --onnx=model/tiny-1920.onnx --saveEngine=model/tiny-1920.trt --fp16 --workspace=5120
```
## Jetson Xavier NX
```shell
# yolov7-w6
sudo /usr/src/tensorrt/bin/trtexec --onnx=model/W6-1920.onnx --saveEngine=model/W6-1920.trt --fp16 --workspace=1000
# yolov7-tiny    We recommend that Jetson Xavier NX use a small model and a small resolution.
sudo /usr/src/tensorrt/bin/trtexec --onnx=model/tiny-960.onnx --saveEngine=model/tiny-960.trt --fp16 --workspace=1000
```
# Build the demo
You should set the eigen path,TensorRT path and CUDA path in CMakeLists.txt.
```shell
cd <MotionTrack_HOME>
mkdir build
cd build
cmake ..
make
```

# Run the demo
you can run the demo with **220 FPS** in Nvidia RTX 3060 Laptop and **40 FPS** in Jetson Xavier NX.
```shell
./build/motiontrack -m model/tiny-960.trt -v res/1_video.avi
```

# Acknowledgements
A large part of the code is borrowed from the previous outstanding work. Many thanks for their wonderful works.
* [https://github.com/ifzhang/ByteTrack/tree/main/deploy/TensorRT/cpp](https://github.com/ifzhang/ByteTrack/tree/main/deploy/TensorRT/cpp)
* [https://github.com/zhiqwang/yolov5-rt-stack/tree/main/deployment/tensorrt](https://github.com/zhiqwang/yolov5-rt-stack/tree/main/deployment/tensorrt)
