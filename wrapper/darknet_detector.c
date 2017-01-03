#include "network.h"
#include "detection_layer.h"
#include "region_layer.h"
#include "cost_layer.h"
#include "utils.h"
#include "parser.h"
#include "box.h"
#include "image.h"
#include "option_list.h"

#include "darknet_detector.h"
//#include <sys/time.h>
// #include <time.h>
// #include <winsock.h>
// #include "gettimeofday.h"

#include "cuda.h"

#define FRAMES 3

#ifdef OPENCV
#include "opencv2/highgui/highgui_c.h"
#include "opencv2/imgproc/imgproc_c.h"
image get_image_from_stream(CvCapture *cap);

static char **demo_names;
static image **demo_alphabet;
static int demo_classes;

static int count;

static int num;
static int num_out = 0;

/* static int running = 0; */

static const char** names_out; 

static process_func_ptr process_func;
static fetch_func_ptr fetch_func;

static float **probs;
static float* probs_out;
static box *boxes;
static box* boxes_out;
static network net;
static image in   ;
static image in_s ;
static image det  ;
static image det_s;
static image disp = {0};
static CvCapture * cap;
// static float fps = 0;
static float demo_thresh = 0;

static pthread_t fetch_thread;
static pthread_t detect_thread;
static pthread_t process_thread;

static float *predictions[FRAMES];
static int demo_index = 0;
static image images[FRAMES];
static float *avg;

static char* demo_prefix;
static int demo_frame_skip;
static char* window_name;

static int delay;

static int visualize = 0;
static int multithread = 0;

static int updatable = 0;

extern void train_detector(char *datacfg, char *cfgfile, char *weightfile, int *gpus, int ngpus, int clear);
extern void validate_detector(char *datacfg, char *cfgfile, char *weightfile);
extern void validate_detector_recall(char *cfgfile, char *weightfile);
extern void test_detector(char *datacfg, char *cfgfile, char *weightfile, char *filename, float thresh);

void *fetch_thread_func(void *ptr)
{
	if (fetch_func) fetch_func(&in);
	/* in = get_image_from_stream(cap); */
	if(!in.data){
		/* error("Stream closed."); */
		error("Invalid image.");
	}
	in_s = resize_image(in, net.w, net.h);
	
	return 0;
}

void *detect_thread_func(void *ptr)
{
	float nms = .4;

	layer l = net.layers[net.n-1];
	float *X = det_s.data;
	float *prediction = network_predict(net, X);

	memcpy(predictions[demo_index], prediction, l.outputs*sizeof(float));
	mean_arrays(predictions, FRAMES, l.outputs, avg);
	l.output = avg;

	free_image(det_s);
	if(l.type == DETECTION){
		get_detection_boxes(l, 1, 1, demo_thresh, probs, boxes, 0);
	} else if (l.type == REGION){
		get_region_boxes(l, 1, 1, demo_thresh, probs, boxes, 0, 0);
	} else {
		error("Last layer must produce detections\n");
	}
	if (nms > 0) do_nms(boxes, probs, num, l.classes, nms);
	// printf("\033[2J");
	// printf("\033[1;1H");
	// printf("\nFPS:%.1f\n",fps);
	// printf("Objects:\n\n");
	
	images[demo_index] = det;
	det = images[(demo_index + FRAMES/2 + 1)%FRAMES];
	demo_index = (demo_index + 1)%FRAMES;

	if (visualize) draw_detections(det, num, demo_thresh, boxes, probs, demo_names, demo_alphabet, demo_classes);

	return 0;
}

void *process_thread_func(void *ptr)
{
	if (process_func) {
		/* if (process_func(num_out, names_out, boxes_out, probs_out)) running = 0; */
		process_func(num_out, names_out, boxes_out, probs_out);
	}
	return 0;
}

/* double get_wall_time()
{
	struct timeval time;
	if (gettimeofday(&time,NULL)){
		return 0;
	}
	return (double)time.tv_sec + (double)time.tv_usec * .000001;
} */

void detector_run(char *cfgfile, char *weightfile, float thresh, int cam_index, const char *filename, char **names, int classes, int frame_skip, char *prefix)
{
	//skip = frame_skip;
	image **alphabet = load_alphabet();
	delay = frame_skip;
	demo_names = names;
	demo_alphabet = alphabet;
	demo_classes = classes;
	demo_thresh = thresh;
	/* printf("Demo\n"); */
	net = parse_network_cfg(cfgfile);
	if(weightfile){
		load_weights(&net, weightfile);
	}
	set_batch_network(&net, 1);

	srand(2222222);
	
	demo_frame_skip = frame_skip;
	demo_prefix = prefix;

	/* if(filename){
		printf("video file: %s\n", filename);
		cap = cvCaptureFromFile(filename);
	}else{
		cap = cvCaptureFromCAM(cam_index);
	}

	if(!cap) error("Couldn't connect to webcam.\n"); */

	layer l = net.layers[net.n-1];
	int j;
	
	num = l.w*l.h*l.n;

	avg = (float *) calloc(l.outputs, sizeof(float));
	for(j = 0; j < FRAMES; ++j) predictions[j] = (float *) calloc(l.outputs, sizeof(float));
	for(j = 0; j < FRAMES; ++j) images[j] = make_image(1,1,3);

	names_out = (char**)calloc(num, sizeof(char*));

	boxes = (box *)calloc(num, sizeof(box));
	boxes_out = (box*)calloc(num, sizeof(box));
	
	probs = (float **)calloc(num, sizeof(float *));
	for(j = 0; j < num; ++j) probs[j] = (float *)calloc(l.classes, sizeof(float *));
	
	probs_out = (float**)calloc(num, sizeof(float*));


	/* fetch_thread_func(0);
	det = in;
	det_s = in_s;

	fetch_thread_func(0);
	detect_thread_func(0);
	disp = det;
	det = in;
	det_s = in_s;

	for(j = 0; j < FRAMES/2; ++j){
		fetch_thread_func(0);
		detect_thread_func(0);
		disp = det;
		det = in;
		det_s = in_s;
	} */
	
	window_name = "Display";

	count = 0;
	if(!prefix && visualize){
		cvNamedWindow(window_name, CV_WINDOW_NORMAL); 
		cvMoveWindow(window_name, 0, 0);
		cvResizeWindow(window_name, 1352, 1013);
	}

	// double before = get_wall_time();
	
	/* running = 1; */

	/* while(running){
		detector_update();
	} */
}

void detector_update(process_func_ptr process_func_in, fetch_func_ptr fetch_func_in)
{
	if (0 == updatable) return;
	
	process_func = process_func_in;
	fetch_func = fetch_func_in;

	printf("frame: %d\n", count);
	++count;
	if(multithread){
		if(pthread_create(&fetch_thread, 0, fetch_thread_func, 0)) error("Thread creation failed");
		if (det.data){
			if(pthread_create(&detect_thread, 0, detect_thread_func, 0)) error("Thread creation failed");
			if(pthread_create(&process_thread, 0, process_thread_func, 0)) error("Thread creation failed");
		}

		pthread_join(fetch_thread, 0);
		if (det.data){
			pthread_join(detect_thread, 0);
			pthread_join(process_thread, 0);
			
			if(delay == 0){
				free_image(disp);
				disp  = det;
			}
			
			num_out = 0;
			
			for(int i = 0; i < num; ++i){
				int class = max_index(probs[i], demo_classes);
				char* name = demo_names[class];
				box bound = boxes[i];
				float prob = probs[i][class];
				if(prob > demo_thresh){
					names_out[num_out] = name;
					boxes_out[num_out] = bound;
					probs_out[num_out] = prob;
					
					++num_out;
				}
			}
			
			if(!demo_prefix){
				if (visualize && disp.data) show_image(disp, window_name);
				int c = cvWaitKey(1);
				if (c == 10){
					if(demo_frame_skip == 0) demo_frame_skip = 60;
					else if(demo_frame_skip == 60) demo_frame_skip = 4;   
					else if(demo_frame_skip == 4) demo_frame_skip = 0;
					else demo_frame_skip = 0;
				}
			}
			else{
				char buff[256];
				sprintf(buff, "%s_%08d", demo_prefix, count);
				save_image(disp, buff);
			}
			
		}

		det   = in;
		det_s = in_s;
	}
	else {
		fetch_thread_func(0);
		
		det   = in;
		det_s = in_s;
		
		detect_thread_func(0);
		
		num_out = 0;
		
		for(int i = 0; i < num; ++i){
			int class = max_index(probs[i], demo_classes);
			char* name = demo_names[class];
			box bound = boxes[i];
			float prob = probs[i][class];
			if(prob > demo_thresh){
				// box.x and box.y is the center position in [0, 1].
				names_out[num_out] = name;
				boxes_out[num_out] = bound;
				probs_out[num_out] = prob;
				
				++num_out;
			}
		}
		
		process_thread_func(0);
		
		if(delay == 0){
			free_image(disp);
			disp  = det;
		}
		
		if(!demo_prefix){
			if (visualize && disp.data) show_image(disp, window_name);
			int c = cvWaitKey(1);
			if (c == 10){
				if(demo_frame_skip == 0) demo_frame_skip = 60;
				else if(demo_frame_skip == 60) demo_frame_skip = 4;   
				else if(demo_frame_skip == 4) demo_frame_skip = 0;
				else demo_frame_skip = 0;
			}
		}
		else{
			char buff[256];
			sprintf(buff, "%s_%08d", demo_prefix, count);
			save_image(disp, buff);
		}

	}

	--delay;
	if(delay < 0){
		delay = demo_frame_skip;

		/* double after = get_wall_time();
		float curr = 1./(after - before);
		fps = curr;
		before = after; */
	}
}

void detector_main(char* module, char* operation, char* datacfg, char* cfg, char* weights, char* filename, char* prefix, float thresh, int frame_skip, char* gpu_list, int clear, int visualize_in, int multithread_in)
{
	/*
	arguments:
		char* module = argv[1];
		char* operation = argv[2];
		char* datacfg = argv[3];
		char* cfg = argv[4];
		char* weights = (argc > 5) ? argv[5] : 0;
		char* filename = (argc > 6) ? argv[6]: 0;
		char* prefix = -prefix, default 0
		float thresh = -thresh, default 0.24
		int cam_index (not used) = -c, default 0
		int frame_skip = -s, default 0
		char* gpu_list = -gpus, default 0
			separated by ,
		bool clear = -clear
		bool visualize = -v
		bool multithread = -m
	*/
	/* char *prefix = find_char_arg(argc, argv, "-prefix", 0); */
	/* float thresh = find_float_arg(argc, argv, "-thresh", .24); */
	/* int cam_index = find_int_arg(argc, argv, "-c", 0); */
	/* int frame_skip = find_int_arg(argc, argv, "-s", 0); */
	visualize = visualize_in;
	multithread = multithread_in;
	/* if(argc < 4){
		fprintf(stderr, "usage: %s %s [train/test/valid] [cfg] [weights (optional)]\n", argv[0], argv[1]);
		return;
	} */
	/* char *gpu_list = find_char_arg(argc, argv, "-gpus", 0); */
	int *gpus = 0;
	int gpu = 0;
	int ngpus = 0;
	if(gpu_list){
		printf("%s\n", gpu_list);
		int len = strlen(gpu_list);
		ngpus = 1;
		int i;
		for(i = 0; i < len; ++i){
			if (gpu_list[i] == ',') ++ngpus;
		}
		gpus = calloc(ngpus, sizeof(int));
		for(i = 0; i < ngpus; ++i){
			gpus[i] = atoi(gpu_list);
			gpu_list = strchr(gpu_list, ',')+1;
		}
	} else {
		gpu = gpu_index;
		gpus = &gpu;
		ngpus = 1;
	}

	/* int clear = find_arg(argc, argv, "-clear"); */

	/* char* operation = argv[1]; */
	/* char* module = argv[2]; */
	/* char *datacfg = argv[3]; */
	/* char *cfg = argv[4]; */
	/* char *weights = (argc > 5) ? argv[5] : 0; */
	/* char *filename = (argc > 6) ? argv[6]: 0; */
	if(0==strcmp(operation, "test")) test_detector(datacfg, cfg, weights, filename, thresh);
	else if(0==strcmp(operation, "train")) train_detector(datacfg, cfg, weights, gpus, ngpus, clear);
	else if(0==strcmp(operation, "valid")) validate_detector(datacfg, cfg, weights);
	else if(0==strcmp(operation, "recall")) validate_detector_recall(cfg, weights);
	else if(0==strcmp(operation, "detect")) {
		list *options = read_data_cfg(datacfg);
		int classes = option_find_int(options, "classes", 20);
		char *name_list = option_find_str(options, "names", "data/names.list");
		char **names = get_labels(name_list);
		updatable = 1;
		detector_run(cfg, weights, thresh, 0, filename, names, classes, frame_skip, prefix);
	}
}

void detector_initialize(int gpu_id)
{
	gpu_index = gpu_id;

#ifndef GPU
	gpu_index = -1;
#else
	if (gpu_index >= 0) {
		cuda_set_device(gpu_index);
	}
#endif
}

#endif

