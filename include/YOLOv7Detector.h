#pragma once
#include <cuda_runtime_api.h>
#include <opencv2/opencv.hpp>
#include <fstream>
#include <iostream>
#include <memory>
#include <string>
#include "NvInfer.h"
#include "NvInferPlugin.h"
#include "NvOnnxParser.h"
#include "cmdline.h"
using namespace nvonnxparser;
using namespace nvinfer1;

#define CHECK(status)                                                      \
    do                                                                     \
    {                                                                      \
        auto ret = (status);                                               \
        if (ret != 0)                                                      \
        {                                                                  \
            std::cerr << __LINE__ << "Cuda failure: " << ret << std::endl; \
            abort();                                                       \
        }                                                                  \
    } while (0)

class MyLogger : public ILogger
{
public:
    MyLogger() = default;
    virtual ~MyLogger() = default;
    void log(Severity severity, AsciiChar const *msg) noexcept override
    {
        // suppress info-level messages
        if (severity <= Severity::kWARNING)
            std::cout << msg << std::endl;
    }
};

struct Detection
{
    cv::Rect box;
    float conf;
    int classId;
};

class YOLOv7Detector
{
public:
    YOLOv7Detector(
        const char *model_path,
        int max_batch_size = 1,
        bool enable_int8 = false,
        bool enable_fp16 = false);
    virtual ~YOLOv7Detector();
    YOLOv7Detector(const YOLOv7Detector &) = delete;
    YOLOv7Detector &operator=(const YOLOv7Detector &) = delete;

    std::vector<Detection> detect(cv::Mat &image);
    std::vector<std::vector<Detection>> detect(std::vector<cv::Mat> &images);

private:
    MyLogger logger;
    std::unique_ptr<ICudaEngine> engine;
    std::unique_ptr<IExecutionContext> context;
    cudaStream_t stream;
}; /* class YOLOv7Detector */