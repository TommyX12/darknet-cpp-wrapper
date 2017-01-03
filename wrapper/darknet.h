#ifndef DARKNET_CPP
#define DARKNET_CPP

extern "C" {
#include "darknet_detector.h"
}

#include <vector>
#include <string>
#include <opencv2/core/core.hpp>

class Darknet
{
	public:
		static Darknet* get_current();
		
		Darknet();
		~Darknet();
		
		void initialize(int gpu_id = 0);
		void load_command_args(int argc, char** argv);
		void load_command_args(const std::vector<std::string>& args);
		void run();
		void process(cv::Mat& image, process_func_ptr process_func = nullptr);
		
		std::string module;
		std::string operation;
		std::string datacfg;
		std::string cfg;
		std::string weights;
		// std::string filename;
		std::string prefix;
		
		float thresh;
		int frame_skip;
		bool visualize;
		bool multithread;
		std::string gpu_list;
		bool clear;
		
	private:
		static Darknet* m_current; 
		
		bool m_initialized, m_running;
		// process_func_ptr m_process_func;
		
		static cv::Mat* m_mat;
		
		static void mat_to_image(image* img);
		
		void validate_initialized();
		void validate_running();
};

#endif
