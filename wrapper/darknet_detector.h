#ifndef DARKNET_DETECTOR
#define DARKNET_DETECTOR

#include "utils.h"
#include "box.h"
#include "image.h"

typedef void (*process_func_ptr)(int num, const char** names, box* boxes, float* probs);
typedef void (*fetch_func_ptr)(image* img);

void detector_process(char *cfgfile, char *weightfile, float thresh, int cam_index, const char *filename, char **names, int classes, int frame_skip, char *prefix);

void detector_update(process_func_ptr process_func_in, fetch_func_ptr fetch_func_in);

void detector_main(char* module, char* operation, char* datacfg, char* cfg, char* weights, char* filename, char* prefix, float thresh, int frame_skip, char* gpu_list, int clear, int visualize_in, int multithread_in);

void detector_initialize(int gpu_id);

#endif
