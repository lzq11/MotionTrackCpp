#include "YOLOv7Detector.h"
#include "MotionTracker.h"
#include "cmdline.h"

int main(int argc, char** argv) {
    
    cmdline::parser cmd;
    cmd.add("int8", '\0', "Enable INT8 inference.");
    cmd.add("fp16", '\0', "Enable FP16 inference.");
    cmd.add<std::string>("model_path", 'm', "Path to TensorRT model.", false, "model/yolov7-tiny.trt");
    cmd.add<std::string>("video", 'v', "video source to be tracked.", false, "res/1_video.avi");

    cmd.parse_check(argc, argv);
    std::string videoPath = cmd.get<std::string>("video");
    std::string modelPath = cmd.get<std::string>("model_path");
    std::vector<std::string> classNames = {"sailing_boat","fishing_boat","floater","passenger_ship","speedboat","cargo","special_ship"};

    // const string videoPath("res/1_video.avi");
    // const string modelPath {"model/yolov7-tiny.trt"};
    // const string modelPath {"model/yolov7-w6.trt"};

    VideoCapture cap(videoPath);
    if (!cap.isOpened())
		return 0;

	int img_w = cap.get(CAP_PROP_FRAME_WIDTH);
	int img_h = cap.get(CAP_PROP_FRAME_HEIGHT);
    int fps = cap.get(CAP_PROP_FPS);
    long nFrame = static_cast<long>(cap.get(CAP_PROP_FRAME_COUNT));
    cout << "Total frames: " << nFrame << endl;

    VideoWriter writer("res/result.mp4", VideoWriter::fourcc('m', 'p', '4', 'v'), fps, Size(img_w, img_h));

    Mat img;

    // YOLOv7Detector detector(modelPath.c_str(), cmd.exist("int8"), cmd.exist("fp16"));
    YOLOv7Detector detector(modelPath.c_str(),1,false,true);//fp16  yolo-w6 3060lap 22fps

    fps = 30; 
    MotionTracker tracker(fps, 30);
    long num_frames = 0;
    long total_ms = 0;
    while (true)
    {
        if(!cap.read(img))
            break;
        num_frames ++;
        if (num_frames % 20 == 0)
        {
            cout << "Processing frame " << num_frames << " (" << num_frames * 1000000 / total_ms << " fps)" << endl;
        }
		if (img.empty())
			break;
        auto start = chrono::system_clock::now();
        std::vector<Detection> objects = detector.detect(img);
        vector<STrack> output_stracks = tracker.update(objects);

        auto end = chrono::system_clock::now();
        total_ms = total_ms + chrono::duration_cast<chrono::microseconds>(end - start).count();

        for (int i = 0; i < output_stracks.size(); i++)
		{
			vector<float> tlwh = output_stracks[i].tlwh;
			// bool vertical = tlwh[2] / tlwh[3] > 1.6;
			if (tlwh[2] * tlwh[3] > 0)
			{
				Scalar s = tracker.get_color(output_stracks[i].track_id);
                putText(img, format("%d %s", output_stracks[i].track_id,classNames[output_stracks[i].classes].c_str()), Point(tlwh[0], tlwh[1] - 5), 0, 0.8, Scalar(0, 0, 255), 2, LINE_AA);
                rectangle(img, Rect(tlwh[0], tlwh[1], tlwh[2], tlwh[3]), s, 4);
			}
		}
        putText(img, format("frame: %d fps: %d num: %d", num_frames, num_frames * 1000000 / total_ms, output_stracks.size()), 
                Point(0, 30), 0, 0.9, Scalar(0, 0, 255), 2, LINE_AA);
        writer.write(img);
        imshow("MotionTrack Tracking Demo",img);
        char c = waitKey(100);
        if (c > 0)
        {
            break;
        }

    }
    cap.release();
    cout << "FPS: " << num_frames * 1000000 / total_ms << endl;
    
}