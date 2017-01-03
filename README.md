# Darknet C++ Wrapper
A small C++ wrapper for pjreddie/darknet detector (yolo v2), for use in UTAT project.

# Installation:
1. Download and make sure you can compile and run darknet yolo. <br/>
	Links: <br/>
		http://pjreddie.com/darknet/yolo/ <br/>
		https://github.com/pjreddie/darknet
	
	Note: OpenCV and GPU(CUDA) must be enabled when compiling. <br/>
	
2. Make a copy of the darknet source code, including the make file. <br/>
	The following steps are performed on this copy, unless stated otherwise.

3. Remove or rename the main function in the darknet source code. <br/>
	There is a main function in the original darknet. In order to use it as a library, this function must be removed or renamed. For example, rename this function to _main instead of main. <br/>
	The function is located in `src/darknet.c`.

4. Copy the content inside `wrapper` folder to somewhere near the source file. <br/>
	This location will need to be manually added to the make file in later steps.
	
5. Add your own C++ file in the same location, include `darknet.h`, and use the `Darknet` class. <br/>
	See `example/main.cpp` for example usage.

6. Add your source code / folder to the make file manually. <br/>
	Make sure to use g++ instead of gcc for compiling the .cpp files.
	
7. Compile and run!

# Reference:
## Darknet class
### Methods:
* static Darknet* get_current() <br/>
	Returns the current instance of darknet. <br/>
	Returns `nullptr` if none are instantiated.
	
* Darknet() <br/>
	Default constructor. <br/>
	The `initialize` method must be called before darknet is used. <br/>
	Note: Only one instance can be constructed.
	
* ~Darknet() <br/>
	Default destructor.

* void initialize(int gpu_id = 0) <br/>
	Initializes darknet with the given `gpu_id`. <br/>
	This method cannot be called more than once.
	
	Parameters:
	* int gpu_id <br/>
		The default GPU id to be used in computation.

* void load_command_args(int argc, char** argv) <br/>
	Sets the object properties of darknet using command line arguments in the command format of the original darknet application.
	
	See original darknet documentation for detail.
		
	Parameters:
	* int argc <br/>
		Number of arguments.
		
	* char** argv <br/>
		list of arguments in the form of an array of C-style strings.

* void load_command_args(const std::vector<std::string>& args) <br/>
	This is an overloaded method. Instead of using C-style command arguments, this method parses a `std::vector` list of C++ strings for convenience.

* void run() <br/>
	Runs the darknet application using the object properties as parameters. <br/>
	Depending on the `operation` property, the behavior of this method may vary. <br/>
	This method cannot be called more than once.

* void process(cv::Mat& image, process_func_ptr process_func = nullptr) <br/>
	If `module` is set to "detector" and `operation` is set to "detect", calling this function will process `image` by detecting objects using darknet's neural network and calling `process_func` with the detected data, including object names, positions and bounding boxes, and confidences (probabilities). <br/>
	If `module` and `operation` is in other configurations, this method has no effect. <br/>
	`run` must be called before this method.
	
	Parameters:
	* cv::Mat& image <br/>
		The OpenCV image to be processed.
		
	* process_func_ptr process_func <br/>
		The callback function to be used in processing the data. <br/>
		Format of the callback function: <br/>
			void process_func(int num, const char** names, box* boxes, float* probs)
		
		Parameters:
		* int num <br/>
			Number of objects detected.
			
		* const char** names <br/>
			Names of the objects detected.
		
		* box* boxes <br/>
			Bounding boxes of the object detected. <br/>
			`box` is a struct with 4 properties:
			* x: x coordinate of the center.
			* y: y coordinate of the center.
			* w: width of the bounding box.
			* h: height of the bounding box.
			
			All properties are in the range of 0.0 to 1.0, where 1.0 is the full width/height of the image.
		
		* float* probs <br/>
			The confidence/probability of the objects detected.
	
### Properties:
* std::string module <br/>
	The darknet module/option to use. <br/>
	Valid values:
	* "detector"
	Corresponds to the first argument in darknet command.
	
* std::string operation <br/>
	The detector operation to use.  <br/>
	Corresponds to the second argument in darknet command. <br/>
	Valid values:
	* "detect"
	* "test"
	* "train"
	* "valid"
	* "recall"

* std::string datacfg <br/>
	The path to the data config file. <br/>
	Example: "cfg/coco.data"
	
* std::string cfg <br/>
	The path to the neural network config file. <br/>
	Example: "cfg/yolo.cfg"

* std::string weights <br/>
	The path to the trained weights file. <br/>
	Example: "yolo.weights"

* std::string prefix <br/>
	If this property is not an empty string, visualization of each frame processed will be saved to hard drive, with this property being the prefix for the name of the files. <br/>
	(Untested)

* float thresh <br/>
	The confidence threshold for detection. Any detected objects with confidence (probably) lower than this property will be discarded.  <br/>
	Default: 0.24.

* int frame_skip <br/>
	(Unknown)

* bool visualize <br/>
	If true, a window will be opened that shows detected objects in each frame processed.

* bool multithread <br/>
	If true, multi-threading will be used to improve performance. <br/>
	Note: this will make each process call processes the second last image passed in. <br/>
	(Experimental)

* std::string gpu_list <br/>
	A list of GPUs to use in computation. Separate the GPU indexes with commas. <br/>
	(Untested)

* bool clear <br/>
	(Unknown)

### Example:
See `example.cpp`.
