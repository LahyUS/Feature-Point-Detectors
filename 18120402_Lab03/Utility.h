#pragma once
#include "opencv2\opencv.hpp"
#include "opencv2\highgui\highgui.hpp"
#include "Convolutioner.h"
#include <vector>
#include "Blob.h"

class Utility
{
public:
	//----------------------------------------------------------------------------------------------------------------------------------

	int LOG(const Mat& sourceImage, Mat& destinationImage, double sigma);

	//----------------------------------------------------------------------------------------------------------------------------------

	Mat scalar_multiplyMatrix(double scalar, Mat matrix);

	//----------------------------------------------------------------------------------------------------------------------------------

	// Correspoding multiply matrix
	Mat multiplyMatrix_Corresponding(Mat a, Mat b);

	//----------------------------------------------------------------------------------------------------------------------------------

	// Multiply matrix normally
	Mat multiplyMatrix(Mat matrix_a, Mat matrix_b);

	//----------------------------------------------------------------------------------------------------------------------------------

	// Sum matrix at each corresponding position 
	Mat sumMatrix_Corresponding(Mat a, Mat b);

	//----------------------------------------------------------------------------------------------------------------------------------

	// Subtract matrix at each corresponding position 
	Mat subMatrix_Corresponding(Mat a, Mat b);

	//----------------------------------------------------------------------------------------------------------------------------------

	// Calculate det of matrix
	Mat det(Mat matrix_a, Mat matrix_b, Mat matrix_c);

	//----------------------------------------------------------------------------------------------------------------------------------

	// Calculate trace of 2 same matrix
	Mat trace(Mat matrix_a, Mat matrix_b);

	//----------------------------------------------------------------------------------------------------------------------------------

	// Find maximum value of matrix
	double max(Mat img);

	//----------------------------------------------------------------------------------------------------------------------------------

	// Find maximum value on a window
	double findMax(Mat window, int x, int y);

	//----------------------------------------------------------------------------------------------------------------------------------

	/* Go through the window, keep the pixel with intensity equal max_Val
	   Set pixel equal zero if it's intensity != max_Val
	*/
	void Nonmax_Suppression(Mat& window, int x, int y, double max_Val);

	//----------------------------------------------------------------------------------------------------------------------------------

	// To get a matrix which each element is squared
	void squareMat(Mat& srcImg);

	//----------------------------------------------------------------------------------------------------------------------------------

	bool removeRedundantBlobs(vector<Blob> list, Blob new_Blob, double ratio);

	//----------------------------------------------------------------------------------------------------------------------------------

	/*
		generate base image
		inputs:
		 - source image of size (r, c).
		outputs:
		 - blurred image of size (2*r, 2*c).
	*/
	Mat generate_base_img(Mat scr);

	//----------------------------------------------------------------------------------------------------------------------------------

	/*
		compute number of octaves
		inputs:
		- size of the image.
		outputs:
		- number of octaves (integer number).
	*/
	int compute_Number_Of_Octaves(Size image_shape);

	//----------------------------------------------------------------------------------------------------------------------------------

	/*
		generate sigma values for blurring
		inputs:
		- sigma value.
		- number of intervals
		outputs:
		- vector of double values.
	*/
	vector<double> generate_sigmas(double sigma, int num_intervals);

	//----------------------------------------------------------------------------------------------------------------------------------

	Mat compute_gradient(Mat current, Mat up, Mat down);

	//----------------------------------------------------------------------------------------------------------------------------------

	Mat compute_Hessian(Mat current, Mat up, Mat down);

	//----------------------------------------------------------------------------------------------------------------------------------

	/*
		check if the center pixel is extremum among its 26 neighbors.
		inputs:
		- current kernel.
		- kernel up.
		- kernel down.
		- threshold value.
		outputs:
		- boolean value indicates if the center pixel is extremum or not.
	*/
	bool is_it_extremum(Mat present_win, Mat pos_win, Mat previous_win, double threshold);

	//----------------------------------------------------------------------------------------------------------------------------------

	/*
		obtain the interpolated estimate of the location of the extremum.
		inputs:
		- coordinates of the extremum (i, j, scale).
		- octave images.
		- octave index.
		outputs:
		- actual extremum point and the corresponding scale.
	*/
	tuple<Point, int> localize_Extremum(int ii, int jj, int sc, vector<Mat> Octave, int octave_index);

	//----------------------------------------------------------------------------------------------------------------------------------

	vector<KeyPoint> remove_duplicate(vector<KeyPoint> keypoints);

	//----------------------------------------------------------------------------------------------------------------------------------

	/*
		create scale space and difference of gaussian images
		inputs:
		- input image.
		- number of intervals.
		- vector of sigma values.
		outputs:
		- tuple of scale space and difference of gaussian images.
	*/
	tuple<vector<vector<Mat>>, vector<vector<Mat>>> Create_Scale_Space(Mat scr, int num_intervals, vector<double> sigmas);

	//----------------------------------------------------------------------------------------------------------------------------------

	vector<tuple<Point, int>> Extrema(vector<vector<Mat>> DoG, vector<vector<Mat>> scale_images, int num_intervals, double sigma);
};

