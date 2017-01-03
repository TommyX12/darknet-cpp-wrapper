# darknet-cpp-wrapper
A small C++ wrapper for pjreddie/darknet detector (yolo v2), for use in UTAT project.

				- Installation:
					1. Download and make sure you can compile and run darknet yolo.
						@ Links:
							http://pjreddie.com/darknet/yolo/
							https://github.com/pjreddie/darknet
						
						Note: OpenCV and GPU(CUDA) must be enabled when compiling.
						
					2. Make a copy of the darknet source code, including the make file.
						The following steps are performed on this copy, unless stated otherwise.
					
					3. Remove or rename the main function in the darknet source code.
						There is a main function in the original darknet. In order to use it as a library, this function must be removed or renamed. For example, rename this function to _main instead of main.
						The function is located in [src/darknet.c].
					
					4. Copy the content inside [wrapper] folder to somewhere near the source file.
						This location will need to be manually added to the make file in later steps.
						
					5. Add your own C++ file in the same location, include [darknet.h], and use the [Darknet] class.
						See [example/main.cpp] for example usage.
					
					6. Add your source code / folder to the make file manually.
						Make sure to use g++ instead of gcc for compiling the .cpp files.
						
					7. Compile and run!

				- Reference:
					. Darknet
						- Methods:
							. static Darknet* get_current()
								Returns the current instance of darknet.
								Returns [nullptr] if none are instantiated.
								
							. Darknet()
								Default constructor.
								The [initialize] method must be called before darknet is used.
								Note: Only one instance can be constructed.
								
							. ~Darknet()
								Default destructor.

							. void initialize(int gpu_id = 0)
								Initializes darknet with the given [gpu_id].
								This method cannot be called more than once.
								
								- Parameters:
									. int gpu_id
										The default GPU id to be used in computation.
							
							. void load_command_args(int argc, char** argv)
								Sets the object properties of darknet using command line arguments in the command format of the original darknet application.
								
								@ See original darknet documentation for detail.
									
								- Parameters:
									. int argc
										Number of arguments.
										
									. char** argv
										list of arguments in the form of an array of C-style strings.
							
							. void load_command_args(const std::vector<std::string>& args)
								This is an overloaded method. Instead of using C-style command arguments, this method parses a [std::vector] list of C++ strings for convenience.
							
							. void run()
								Runs the darknet application using the object properties as parameters.
								Depending on the [operation] property, the behavior of this method may vary.
								This method cannot be called more than once.
							
							. void process(cv::Mat& image, process_func_ptr process_func = nullptr)
								If [module] is set to "detector" and [operation] is set to "detect", calling this function will process [image] by detecting objects using darknet's neural network and calling [process_func] with the detected data, including object names, positions and bounding boxes, and confidences (probabilities).
								If [module] and [operation] is in other configurations, this method has no effect.
								[run] must be called before this method.
								
								- Parameters:
									. cv::Mat& image
										The OpenCV image to be processed.
										
									. process_func_ptr process_func
										The callback function to be used in processing the data.
										Format of the callback function:
											void process_func(int num, const char** names, box* boxes, float* probs)
										
										Parameters:
											. int num
												Number of objects detected.
												
											. const char** names
												Names of the objects detected.
											
											. box* boxes
												Bounding boxes of the object detected.
												[box] is a struct with 4 properties:
													. x:
														x coordinate of the center.
														
													. y:
														y coordinate of the center.
														
													. w:
														width of the bounding box.
													
													. h:
														height of the bounding box.
												
												All properties are in the range of 0.0 to 1.0, where 1.0 is the full width/height of the image.
											
											. float* probs
												The confidence/probability of the objects detected.
							
						- Properties:
							. std::string module
								The darknet module/option to use. 
								Valid values:
									. "detector"
								Corresponds to the first argument in darknet command.
								
							. std::string operation
								The detector operation to use. 
								Corresponds to the second argument in darknet command.
								Valid values:
									. "detect"
									. "test"
									. "train"
									. "valid"
									. "recall"
							
							. std::string datacfg
								The path to the data config file.
								Example: "cfg/coco.data"
								
							. std::string cfg
								The path to the neural network config file.
								Example: "cfg/yolo.cfg"
							
							. std::string weights
								The path to the trained weights file.
								Example: "yolo.weights"
							
							. std::string prefix
								If this property is not an empty string, visualization of each frame processed will be saved to hard drive, with this property being the prefix for the name of the files.
								(Untested)
							
							. float thresh
								The confidence threshold for detection. Any detected objects with confidence (probably) lower than this property will be discarded. 
								Default: 0.24.
							
							. int frame_skip
								(Unknown)
							
							. bool visualize
								If true, a window will be opened that shows detected objects in each frame processed.
							
							. bool multithread
								If true, multi-threading will be used to improve performance.
								Note: this will make each process call processes the second last image passed in.
								(Experimental)
							
							. std::string gpu_list
								A list of GPUs to use in computation. Separate the GPU indexes with commas.
								(Untested)
							
							. bool clear
								(Unknown)
						
						- Example:
							@ See [example.cpp].
