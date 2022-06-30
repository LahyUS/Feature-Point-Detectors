#include "Point_Detector.h"
#include "Convolutioner.h"
#include "opencv2/opencv.hpp "
#include "opencv2/highgui/highgui.hpp"
#include "Utility.h"
#include <math.h>
#include <vector>
#include "Blob.h"
#include <string>
using namespace cv;

/************************************* MAIN OPERATOR **********************************************/

// Point Detection using Harris's Algorithm
Mat Point_Detector::detectHarrist(Mat img)
{
	Utility util;
	Mat dst_img;
	double sigma = 2;
	dst_img.create(img.rows, img.cols, img.type());
	
	// Firstly, convert source image into grayscale image
	Mat clone_img = img.clone();
	Mat gray_img;
	cv::cvtColor(clone_img, gray_img, cv::COLOR_BGR2GRAY);

	// Then, clarify noise using Gaussian filter
	Mat blur_img = Mat(gray_img.rows, gray_img.cols, gray_img.type());
	cv::GaussianBlur(gray_img, blur_img, cv::Size(5, 5), sigma);

	/* Compute the horizontal and vertical derivatives of the image Ix and Iy by convolving the original image with derivatives of Sobel */
	Mat I_X, I_Y;
	cv::Sobel(blur_img, I_X, CV_64F, 1, 0, 5);
	cv::Sobel(blur_img, I_Y, CV_64F, 0, 1, 5);

	/* Compute the three images corresponding to the outer products of these gradients.(The matrix A is symmetric, so only three entries are needed.) */
	Mat A = util.multiplyMatrix_Corresponding(I_X, I_X);
	Mat B = util.multiplyMatrix_Corresponding(I_X, I_Y);
	Mat C = util.multiplyMatrix_Corresponding(I_Y, I_Y);

	// Convolve each of these images with a larger Gaussian.
	Mat A_, B_, C_;
	cv::GaussianBlur(A, A_, cv::Size(5, 5), sigma);
	cv::GaussianBlur(B, B_, cv::Size(5, 5), sigma);
	cv::GaussianBlur(C, C_, cv::Size(5, 5), sigma);

	// Compute a scalar interest measure using one of the formulas discussed above.
	double alpha = 0.04;
	Mat DET = util.det(A_, B_, C_);
	Mat TRACE = util.trace(A_, B_);
	Mat Alpha_Trace = alpha * TRACE;
	Mat R = util.subMatrix_Corresponding(DET, Alpha_Trace);

	//Find local maxima above a certain threshold and report them as detected feature point locations.
	double max_Val = util.max(R);
	double threshhold = max_Val / 500;
	Mat R_thresh;
	cv::threshold(R, R_thresh, threshhold, max_Val, cv::THRESH_BINARY);

	// Non-maxima suppression
	int height = R_thresh.rows;
	int width = R_thresh.cols;
	for (int i = 1; i < height - 1; i += 2)
	{
		for (int j = 1; j < width - 1; j += 2)
		{
			double max_pixel = util.findMax(R_thresh, i, j);
			util.Nonmax_Suppression(R_thresh, i, j, max_pixel);
		}
	}

	Mat R_dst = cv::Mat(R_thresh.rows, R_thresh.cols, R_thresh.type());
	cv::dilate(R_thresh, R_dst, cv::CAP_ANY);

	// Show feature point on image
	Mat show_on_img = img.clone();
	for (int i = 0; i <= R_dst.rows - 1; i += 1)
	{
		for (int j = 0; j <= R_dst.cols - 1; j += 1)
		{
			double point_value = R_dst.at<double>(i, j);
			if (point_value != 0)
			{
				show_on_img.at<cv::Vec3b>(i, j)[0] = 0;
				show_on_img.at<cv::Vec3b>(i, j)[1] = 0;
				show_on_img.at<cv::Vec3b>(i, j)[2] = 255;
			}
		}
	}

	return show_on_img;
}


// Point Detection using Blob's Algorithm
Mat Point_Detector::detectBlob(Mat srcImg)
{
	// Convert source image to grayscale image
	Mat gray_img;
	cv::cvtColor(srcImg, gray_img, cv::COLOR_BGR2GRAY);

	// Initialization
	Convolutioner conv;
	Utility util;
	double t = 1.414;
	double sigma = 1.0;
	int n_LOG = 5; // Number of LOG image

	// Get Laplacian of Gassian image list
	Mat* LOG_img = new Mat[n_LOG];
	for (int i = 0; i < n_LOG; i++)
	{
		double y = pow(t, i);
		double sigma_i = sigma * y; // sigma_i = sigma * (1.414 ^ i) = 1.0 * (1.414 ^ i)
		util.LOG(gray_img, LOG_img[i], sigma_i);
		util.squareMat(LOG_img[i]);
	}

	// Detect blobs
	vector<Blob> blob_list;
	int height = LOG_img[0].rows;
	int width = LOG_img[0].cols;
	for (int i = 1; i < height - 1; i++)
	{
		for (int j = 1; j < width - 1; j++)
		{
			int filter_k_th = 0;
			int x = -1, y = -1;
			uchar max = 0;
			// Find maxima on sliding window of each filter k'th
			for (int k = 0; k < n_LOG; k++)
			{
				// Consider sliding window 3x3 at present position i,j
				for (int l = -1; l <= 1; l++)
				{
					for (int m = -1; m <= 1; m++)
					{
						uchar intensity = LOG_img[k].at<uchar>(i + l, j + m);
						if (max < intensity)
						{
							max = intensity;
							x = l;
							y = m;
							filter_k_th = k;
						}
					}
				}
			}
			if (max >= 0.03) // Threshold. This threshold is found by trial and error method.
			{
				double radius = pow(t, filter_k_th * sigma) * 3.14;
				Blob new_blob(i + x, j + y, radius);
				blob_list.push_back(new_blob);
			}
		}
	}
	
	// Show blobs on image
	Mat show_on_img = srcImg.clone();
	for (int i = 0; i < blob_list.size(); i += 1)
	{
		int x = blob_list[i].getX();
		int y = blob_list[i].getY();
		double radius = blob_list[i].getR();
		cv::circle(show_on_img, Point(y, x), radius, (0, 0, 255, 0), 0.5);
	}	return show_on_img;
}


// Point Detection using DOG Algorithm
Mat Point_Detector::detectDOG(Mat srcImg)
{
	Mat float_Img, gray_img, base_img;
	// Convert source image to grayscale image
	cv::cvtColor(srcImg, gray_img, cv::COLOR_BGR2GRAY);
	// Convert source image to float image
	gray_img.convertTo(float_Img, CV_32FC1);

	int num_intervals = 2;
	double sigma = 1.0;
	Utility util;

	// Create base image (double size)
	base_img = util.generate_base_img(float_Img);

	// Create sigma list for multiple scale of sigma
	vector<double> sigmas = util.generate_sigmas(sigma, num_intervals);

	// Create a space scale image and DOG images correspoding
	tuple<vector<vector<Mat>>, vector<vector<Mat>>> Scale_DoG = util.Create_Scale_Space(base_img, num_intervals, sigmas);

	// Find the local extremas on DOG image, which are considered as feature points
	vector<tuple<Point, int>> keypoints = util.Extrema(get<1>(Scale_DoG), get<0>(Scale_DoG), num_intervals, sigma);
	
	// Show feature points on image
	Mat show_on_img;
	resize(srcImg, show_on_img, Size(0, 0), 2, 2, INTER_LINEAR);

	// Visualize keypoint
	double k = pow(2, (1. / num_intervals));
	for (int i = 0; i < keypoints.size(); i += 1)
	{
		Point p = get<0>(keypoints[i]);
		int scale = get<1>(keypoints[i]);
		float radius = pow(k, scale) * 3;
		int X = p.x;
		int Y = p.y;

		// Fill circle with a specific radius for visibility
		cv::circle(show_on_img, Point(X, Y), radius, (0, 150, 255, 0), FILLED);
	}

	return show_on_img;
}


// Matching 2 image base on SIFT feature with KNN algorithm
double Point_Detector::matchBySIFT(Mat img1, Mat img2, int detector)
{
	return 0;
}
