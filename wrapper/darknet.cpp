#include "darknet.h"

#include <stdio.h>
#include <string.h>

#include <iostream>


void show_error(char* msg)
{
	std::cerr << msg << std::endl;
	throw msg;
}

char* c_str_copy(const std::string& str)
{
	if (!str.size()) return nullptr;
	
	char* result = new char[str.size() + 1];
	strcpy(result, str.c_str());
	return result;
}


Darknet* Darknet::m_current = nullptr;
cv::Mat* Darknet::m_mat = nullptr;

Darknet* Darknet::get_current()
{
	return Darknet::m_current;
}

Darknet::Darknet()
{
	if (Darknet::m_current == nullptr){
		Darknet::m_current = this;
		
		m_initialized = false;
		
		this->module = "detector";
		this->thresh = 0.24f;
		this->frame_skip = 0;
		this->visualize = false;
		this->multithread = false;
		this->clear = false;
	}
	else {
		show_error("Darknet cannot be instantiated more than once.");
	}
}

Darknet::~Darknet()
{
	
}

void Darknet::initialize(int gpu_id)
{
	if (m_initialized) show_error("Darknet cannot be initialized more than once.");
	m_initialized = true;
	
	detector_initialize(gpu_id);
}

void Darknet::validate_initialized()
{
	if (!m_initialized) show_error("Darknet needs to be initialized before use.");
}

void Darknet::validate_running()
{
	if (!m_running) show_error("Darknet needs to be running.");
}

void Darknet::load_command_args(int argc, char** argv)
{
	validate_initialized();
	
	if (argc < 2) {
		fprintf(stderr, "usage: %s <function>\n", argv[0]);
		return;
	}
	if(argc < 4){
		fprintf(stderr, "detector usage: %s %s [train/test/valid] [cfg] [weights (optional)]\n", argv[0], argv[1]);
		return;
	}
	
	// TODO what happens when std::string is met with a nullptr?
	// std::cout << "hello" << std::endl;
	this->prefix = std::string(find_char_arg(argc, argv, "-prefix", ""));
	this->thresh = find_float_arg(argc, argv, "-thresh", .24);
	this->frame_skip = find_int_arg(argc, argv, "-s", 0);
	this->visualize = find_arg(argc, argv, "-v");
	this->multithread = find_arg(argc, argv, "-m");
	this->gpu_list = std::string(find_char_arg(argc, argv, "-gpus", ""));
	this->clear = find_arg(argc, argv, "-clear");
	
	this->module = argv[1];
	if (this->module == "detector"){
		this->operation = argv[2];
		this->datacfg = argv[3];
		this->cfg = argv[4];
		this->weights = (argc > 5 && argv[5]) ? argv[5] : "";
		// this->filename = (argc > 6 && argv[6]) ? argv[6] : "";
	}
	else {
		fprintf(stderr, "Not an option: %s\n", this->module.c_str());
	}
}

void Darknet::load_command_args(const std::vector<std::string>& args)
{
	int argc = args.size();
	char** argv = new char*[argc];
	
	for (int i = 0; i < argc; ++i){
		const std::string& str = args[i];
		
		char* arg = new char[str.size() + 1];
		strcpy(arg, str.c_str());
		
		argv[i] = arg;
	}
	
	load_command_args(argc, argv);
	
	for (int i = 0; i < argc; ++i){
		delete argv[i];
	}
	delete[] argv;
}

void Darknet::run()
{
	validate_initialized();

	if (m_running) show_error("Darknet is already running.");
	m_running = true;
	
	if (this->module == "detector") {
		// TODO: make temp c_str from c++ string
		char* module = c_str_copy(this->module);
		char* operation = c_str_copy(this->operation);
		char* datacfg = c_str_copy(this->datacfg);
		char* cfg = c_str_copy(this->cfg);
		char* weights = c_str_copy(this->weights);
		// char* filename = c_str_copy(this->filename);
		char* prefix = c_str_copy(this->prefix);
		char* gpu_list = c_str_copy(this->gpu_list);
			
		detector_main(module, operation, datacfg, cfg, weights, 0, prefix, this->thresh, this->frame_skip, gpu_list, this->clear, this->visualize, this->multithread);
		
		// TODO: delete those temp stuffs OR NOT CUZ WE MIGHT NEED IT
		/* delete module;
		delete operation;
		delete datacfg;
		delete cfg;
		delete weights;
		delete filename;
		delete prefix;
		delete gpu_list; */
	}
	else {
		fprintf(stderr, "Not an option: %s\n", this->module.c_str());
	}
}

void Darknet::mat_to_image(image* img)
{
	if (!m_mat) return;
	
	CV_Assert(m_mat->depth() == CV_8U);
	
	const int h = m_mat->rows;
	const int w = m_mat->cols;
	const int channels = m_mat->channels();
	
	*img = make_image(w, h, 3);
	
	int count = 0;

	switch(channels){
		case 1:{
			cv::MatIterator_<unsigned char> it, end;
			for (it = m_mat->begin<unsigned char>(), end = m_mat->end<unsigned char>(); it != end; ++it){
				img->data[count] = img->data[w*h + count] = img->data[w*h*2 + count] = (float)(*it)/255.0;
				
				++count;
			}
			break;
		}
			
		case 3:{
			cv::MatIterator_<cv::Vec3b> it, end;
			for (it = m_mat->begin<cv::Vec3b>(), end = m_mat->end<cv::Vec3b>(); it != end; ++it){
				img->data[count] = (float)(*it)[2]/255.0;
				img->data[w*h + count] = (float)(*it)[1]/255.0;
				img->data[w*h*2 + count] = (float)(*it)[0]/255.0;
				
				++count;
			}
			break;
		}
			
		default:
			show_error("Channel number not supported.");
			break;
	}
}

void Darknet::process(cv::Mat& image, process_func_ptr process_func)
{
	validate_initialized();
	validate_running();
	
	m_mat = &image;
	
	detector_update(process_func, mat_to_image);
}
