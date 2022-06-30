#pragma once
#include "opencv2/opencv.hpp "
#include "opencv2/highgui/highgui.hpp"
using namespace cv;
using namespace std;

class Point_Detector
{
public:
	/* Point Detection using Harris's Algorithm
	   Find feature points of input image, then mark these point on output image
	   Input: Source image	
	   Output: Image marked with feature point
	*/
	Mat detectHarrist(Mat img);


	/* Point Detection using Blob's Algorithm
	   Find feature points of input image, then mark these point on output image
	   Input: Source image
	   Output: Image marked with feature point
	*/
	Mat detectBlob(Mat img);


	/* Point Detection using DOG Algorithm
	   Find feature points of input image, then mark these point on output image
	   Input: Source image
	   Output: Image marked with feature point
	*/
	Mat detectDOG(Mat img);


	/* Matching 2 image base on SIFT feature with KNN algorithm 
	   Input: 
			+ 2 Source image
			+ detector: one of 3 algorithm above (Harris, Blob, DOG)
	   Output: Image matched with feature point from 2 source image
	*/
	double matchBySIFT(Mat img1, Mat img2, int detector);

	// Constructor
	Point_Detector() {}
	// Destructor
	~Point_Detector() {}
};

