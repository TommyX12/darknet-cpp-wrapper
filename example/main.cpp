#include "darknet.h"

#include <iostream>

#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>

using namespace std;
using namespace cv;

VideoCapture camera;

void process_func(int num, const char** names, box* boxes, float* probs)
{
	for (int i = 0; i < num; ++i){
		const char* name = names[i];
		box bound = boxes[i];
		float prob = probs[i];
		printf("%s \t%f,%f,%f,%f \t%f\n", name, bound.x, bound.y, bound.w, bound.h, prob);
	}
}

Darknet darknet;

int main(int argc, char **argv)
{
	Mat img, img_out;
	
	// - Setup sharpening kernel
	Mat sharpen_kernel;
	sharpen_kernel = (Mat_<float>(3, 3) <<	0, -1, 0,
											-1, 5, -1,
											0, -1, 0);
	
	// - Open the default camera
	camera.open(0); 
	if (!camera.isOpened()) throw "Camera cannot be opened.";
	
	// - Initialize darknet
	darknet.initialize(0);
	
	// - Using vector to load commands:
	/* 
	vector<string> args = {"darknet", "detector", "detect", "-v", "cfg/coco.data", "cfg/yolo.cfg", "yolo.weights"};
	darknet.load_command_args(args);
	 */
	
	// - Manually setting properties:
	/* 
	darknet.module = "detector";
	darknet.operation = "detect";
	darknet.visualize = true;
	darknet.datacfg = "cfg/coco.data";
	darknet.cfg = "cfg/yolo.cfg";
	darknet.weights = "yolo.weights";
	 */
	
	// - Load commands with traditional command line arguments format:
	darknet.load_command_args(argc, argv);
	darknet.run();
	
	while(true){
		// - Read from camera
		camera >> img;
		
		// - Sharpen the image with OpenCV
		filter2D(img, img_out, img.depth(), sharpen_kernel);
		
		// - Process the image with darknet
		darknet.process(img_out, process_func);
		
		waitKey(30);
	}
	
	return 0;
}

