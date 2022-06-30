#include "Utility.h"

#define PI 3.14;


int Utility::LOG(const Mat& sourceImage, Mat& destinationImage, double sigma)
{
	if (sourceImage.empty() == true)
		return 0;
	Convolutioner conv;
	vector<double> kernel;

	// LoG function takes sigma as inputand generates a filter of size 6 * sigma(best option).
	int ksize = (int)ceil(6 * sigma);
	int hafl_W, hafl_H;

	//create kernel for LOG
	hafl_W = ksize / 2;
	hafl_H = ksize / 2;

	double a = -(1 / (3.14 * pow(sigma, 4)));
	// calculate kernel 
	for (int y = -hafl_H; y <= hafl_H; y++)
	{
		for (int x = -hafl_W; x <= hafl_W; x++)
		{
			double b = 1 - ((x * x + y * y) / (2 * sigma * sigma));
			double e = expf(-(y * y + x * x) / (2 * sigma * sigma));
			double LOG = a * b * e * pow(sigma, 2);
			kernel.push_back(LOG);
		}
	}

	conv.setKernel(kernel, ksize, ksize);
	return conv.operateConvolution(sourceImage, destinationImage);
}

//-------------------------------------------------------------------------------

Mat Utility::scalar_multiplyMatrix(double scalar, Mat matrix)
{
	Mat result = Mat(matrix.rows, matrix.cols, CV_64F);
	for (int i = 0; i < matrix.rows; i++)
	{
		for (int j = 0; j < matrix.cols; j++)
		{
			double present_value = matrix.at<double>(i, j);
			result.at<double>(i, j) = present_value * scalar;
		}
	}

	return result;
}

// Correspoding multiply matrix
Mat Utility::multiplyMatrix_Corresponding(Mat a, Mat b)
{
	Mat result = Mat(a.rows, a.cols, CV_64F);
	for (int i = 0; i < a.rows; i++)
	{
		for (int j = 0; j < a.cols; j++)
		{
			double value_a = a.at<double>(i, j);
			double value_b = b.at<double>(i, j);
			result.at<double>(i, j) = value_a * value_b;
		}
	}

	return result;
}

//-------------------------------------------------------------------------------

// Multiply matrix normally
Mat Utility::multiplyMatrix(Mat matrix_a, Mat matrix_b)
{
	int h1, w1, h2, w2;
	h1 = matrix_a.rows; w1 = matrix_a.cols;
	h2 = matrix_b.rows; w2 = matrix_b.cols;
	Mat result = Mat(h1, w2, CV_64F);

	if (w1 != h2)
		return result;

	for (int i = 0; i < h1; i++)
	{
		for (int j = 0; j < w2; j++)
		{
			double sum = 0;
			for (int k = 0; k < w1; k++)
			{
				double a_ik = matrix_a.at<double>(i, k);
				double b_kj = matrix_b.at<double>(k, j);
				sum += a_ik * b_kj;
			}

			result.at<double>(i, j) = sum;
		}
	}

	return result;
}

//-------------------------------------------------------------------------------

// Sum matrix at each corresponding position 
Mat Utility::sumMatrix_Corresponding(Mat a, Mat b)
{
	Mat result = Mat(a.rows, a.cols, CV_64F);
	for (int i = 0; i < a.rows; i++)
	{
		for (int j = 0; j < a.cols; j++)
		{
			double value_a = a.at<double>(i, j);
			double value_b = b.at<double>(i, j);
			result.at<double>(i, j) = value_a - value_b;
		}
	}

	return result;
}

//-------------------------------------------------------------------------------

// Subtract matrix at each corresponding position 
Mat Utility::subMatrix_Corresponding(Mat a, Mat b)
{
	Mat result = Mat(a.rows, a.cols, CV_64F);
	for (int i = 0; i < a.rows; i++)
	{
		for (int j = 0; j < a.cols; j++)
		{
			double value_a = a.at<double>(i, j);
			double value_b = b.at<double>(i, j);
			result.at<double>(i, j) = value_a - value_b;
		}
	}

	return result;
}

//-------------------------------------------------------------------------------

// Calculate det of matrix
Mat Utility::det(Mat matrix_a, Mat matrix_b, Mat matrix_c)
{
	Mat a_mul_b = multiplyMatrix_Corresponding(matrix_a, matrix_b);
	Mat c_squared = multiplyMatrix_Corresponding(matrix_c, matrix_c);
	Mat result = subMatrix_Corresponding(a_mul_b, c_squared);
	return result;
}

//-------------------------------------------------------------------------------

// Calculate trace of 2 same matrix
Mat Utility::trace(Mat matrix_a, Mat matrix_b)
{
	Mat result = sumMatrix_Corresponding(matrix_a, matrix_b);
	return result;
}

//-------------------------------------------------------------------------------

// Find maximum value of matrix
double Utility::max(Mat img)
{
	double max = 0;
	for (int i = 0; i < img.rows; i++)
	{
		for (int j = 0; j < img.cols; j++)
		{
			double present_val = img.at<double>(i, j);
			if (max < present_val)
				max = present_val;
		}
	}

	return max;
}

//-------------------------------------------------------------------------------

// Find maximum value on a window
double Utility::findMax(Mat window, int x, int y)
{
	double max = 0;
	for (int i = -1; i <= 1; i++)
	{
		for (int j = -1; j <= 1; j++)
		{
			double present_val = window.at<double>(x + i, y + j);
			if (max < present_val)
				max = present_val;
		}
	}

	return max;
}

//-------------------------------------------------------------------------------

/* Go through the window, keep the pixel with intensity equal max_Val
   Set pixel equal zero if it's intensity != max_Val
*/
void Utility::Nonmax_Suppression(Mat& window, int x, int y, double max_Val)
{
	for (int i = -1; i <= 1; i++)
	{
		for (int j = -1; j <= 1; j++)
		{
			double present_val = window.at<double>(x + i, y + j);
			if (present_val == max_Val)
				continue;
			else
				window.at<double>(x + i, y + j) == 0;
		}
	}
}

//-------------------------------------------------------------------------------

// To get a matrix which each element is squared
void Utility::squareMat(Mat& srcImg)
{
	for (int i = 0; i < srcImg.rows; i++)
	{
		for (int j = 0; j < srcImg.cols; j++)
		{
			uchar intensity = srcImg.at<uchar>(i, j);
			uchar square = intensity * intensity;
			srcImg.at<uchar>(i, j) = square;
		}
	}
}

//-------------------------------------------------------------------------------

bool Utility::removeRedundantBlobs(vector<Blob> list, Blob new_Blob, double ratio)
{
	bool flag = false;
	Blob partner();
	double min_distance = 1000;
	int k;

	// Find the closest blobs
	for (int i = 0; i < list.size(); i++)
	{
		double present_distance = sqrt(pow(new_Blob.getX() - list[i].getX(), 2) + pow(new_Blob.getY() - list[i].getY(), 2));
		if (present_distance < min_distance)
		{
			min_distance = present_distance;
			k = i;
		}
	}
	return flag;
}

//-------------------------------------------------------------------------------

/*
	generate base image
	inputs:
	 - source image of size (r, c).
	outputs:
	 - blurred image of size (2*r, 2*c).
*/
Mat Utility::generate_base_img(Mat scr)
{
	Mat outImg;
	// Blur the image firstly
	GaussianBlur(scr, outImg, Size(0, 0), 0.5, 0.5);

	// Resize image with a double size
	resize(outImg, outImg, Size(0, 0), 2, 2, INTER_LINEAR);
	return outImg;
}

//-------------------------------------------------------------------------------

/*
	compute number of octaves
	inputs:
	- size of the image.
	outputs:
	- number of octaves (integer number).
*/
int Utility::compute_Number_Of_Octaves(Size image_shape)
{
	return int(round(log(min(image_shape.height, image_shape.width)) / log(2) - 1));
}

//-------------------------------------------------------------------------------
/*
	generate sigma values for blurring
	inputs:
	- sigma value.
	- number of intervals
	outputs:
	- vector of double values.
*/
vector<double> Utility::generate_sigmas(double sigma, int num_intervals)
{

	double new_sigma;
	int num_images_per_octave = num_intervals + 3;
	double k = pow(2, (1. / num_intervals));
	vector<double> sigmas;
	sigmas.push_back(sigma);

	for (int i = 1; i < num_images_per_octave; i++)
	{
		new_sigma = pow(k, i) * sigma;
		sigmas.push_back(new_sigma);
	}

	return sigmas;
}

//-------------------------------------------------------------------------------

Mat Utility::compute_gradient(Mat current, Mat up, Mat down)
{
	double dx, dy, dsigma;
	dx = 0.5 * (current.at<float>(1, 2) - current.at<float>(1, 0));
	dy = 0.5 * (current.at<float>(2, 1) - current.at<float>(0, 1));
	dsigma = 0.5 * (up.at<float>(1, 1) - down.at<float>(1, 1));
	Mat gradient = (Mat_<float>(3, 1) << dx, dy, dsigma);

	return gradient;
}

//-------------------------------------------------------------------------------

/*
	Compute Hessian matrix
*/
Mat Utility::compute_Hessian(Mat current, Mat up, Mat down)
{
	double dxx, dyy, dss, dxy, dxs, dys;
	dxx = current.at<float>(1, 2) - 2 * current.at<float>(1, 1) + current.at<float>(1, 0);
	dyy = current.at<float>(2, 1) - 2 * current.at<float>(1, 1) + current.at<float>(0, 1);
	dss = up.at<float>(1, 1) - 2 * current.at<float>(1, 1) + down.at<float>(1, 1);
	dxy = 0.25 * (current.at<float>(2, 2) - current.at<float>(0, 2) - current.at<float>(2, 0) + current.at<float>(0, 0));
	dxs = 0.25 * (up.at<float>(1, 2) - down.at<float>(1, 2) - up.at<float>(1, 0) + down.at<float>(1, 0));
	dys = 0.25 * (up.at<float>(2, 1) - down.at<float>(2, 1) - up.at<float>(0, 1) + down.at<float>(0, 1));
	Mat Hessian = (Mat_<float>(3, 3) << dxx, dxy, dxs,
		dxy, dyy, dys,
		dxs, dys, dss);
	return Hessian;
}

//-------------------------------------------------------------------------------

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
bool Utility::is_it_extremum(Mat present_win, Mat pos_win, Mat previous_win, double threshold)
{
	float center_pixel = present_win.at<float>(1, 1);
	if (abs(center_pixel) > threshold)
	{
		MatConstIterator_<float> it_curr = present_win.begin<float>(), it_curr_end = present_win.end<float>();
		MatConstIterator_<float> it_curr_before_center = next(present_win.begin<float>(), 4);
		MatConstIterator_<float> it_curr_after_center = next(present_win.begin<float>(), 5);

		MatConstIterator_<float> it_up = pos_win.begin<float>(), it_up_end = pos_win.end<float>();

		MatConstIterator_<float> it_down = previous_win.begin<float>(), it_down_end = previous_win.end<float>();

		if (all_of(it_up, it_up_end, [center_pixel](float i) {return center_pixel > i; })
			&& all_of(it_down, it_down_end, [center_pixel](float i) {return center_pixel > i; })
			&& all_of(it_curr, it_curr_before_center, [center_pixel](float i) { return center_pixel > i; })
			&& all_of(it_curr_after_center, it_curr_end, [center_pixel](float i) {return center_pixel > i; }))
		{
			return true;
		}
		else if (all_of(it_up, it_up_end, [center_pixel](float i) {return center_pixel < i; })
			&& all_of(it_down, it_down_end, [center_pixel](float i) {return center_pixel < i; })
			&& all_of(it_curr, it_curr_before_center, [center_pixel](float i) {return center_pixel < i; })
			&& all_of(it_curr_after_center, it_curr_end, [center_pixel](float i) {return center_pixel < i; }))
		{
			return true;
		}

	}
	return false;
}

//-------------------------------------------------------------------------------

/*
	obtain the interpolated estimate of the location of the extremum.
	inputs:
	- coordinates of the extremum (i, j, scale).
	- octave images.
	- octave index.
	outputs:
	- actual extremum point and the corresponding scale.
*/
tuple<Point, int> Utility::localize_Extremum(int ii, int jj, int sc, vector<Mat> Octave, int octave_index)
{
	Point P;
	Mat gradient, Hessian, extremum_update;
	Mat current_img, img_up, img_down;
	int previous_scale = -1;
	int attempt;
	for (attempt = 0; attempt < 5; attempt++)
	{
		Octave[sc].copyTo(current_img);
		Octave[sc + 1].copyTo(img_up);
		Octave[sc - 1].copyTo(img_down);
		// normalize images into range [0-1]
		if (previous_scale != sc)
		{
			previous_scale = sc;
			normalize(current_img, current_img, 0, 1, NORM_MINMAX);
			normalize(img_up, img_up, 0, 1, NORM_MINMAX);
			normalize(img_down, img_down, 0, 1, NORM_MINMAX);
		}
		// compute gradient 
		gradient = compute_gradient(current_img(Rect(jj - 1, ii - 1, 3, 3)),
			img_up(Rect(jj - 1, ii - 1, 3, 3)), img_down(Rect(jj - 1, ii - 1, 3, 3)));
		// compute Hessian matrix
		Hessian = compute_Hessian(current_img(Rect(jj - 1, ii - 1, 3, 3)),
			img_up(Rect(jj - 1, ii - 1, 3, 3)), img_down(Rect(jj - 1, ii - 1, 3, 3)));
		// compute the location of the extremum
		solve(Hessian, gradient, extremum_update);
		extremum_update = -extremum_update;
		// stop if the offset is less than 0.5 in all of its three dimensions
		if (abs(extremum_update.at<float>(0)) < 0.5 && abs(extremum_update.at<float>(1)) < 0.5 && abs(extremum_update.at<float>(2)) < 0.5)
		{
			break;
		}
		// update coordinates
		ii += int(round(extremum_update.at<float>(1)));
		jj += int(round(extremum_update.at<float>(0)));
		sc += int(round(extremum_update.at<float>(2)));
		//check if extremum is outside the image
		if (ii < 5 || ii > current_img.size().height - 5 || jj < 5 || jj > current_img.size().width - 5 || sc < 1 || sc >= Octave.size() - 1)
		{
			P.x = -1;
			P.y = -1;
			return make_tuple(P, -1);
		}
	}
	if (attempt >= 4)
	{
		P.x = -1;
		P.y = -1;
		return make_tuple(P, -1);
	}
	// elemnating low contrast
	if (abs(current_img.at<float>(ii, jj) < 0.03)) // CONTRAST_THRESHOLD = 0.03
	{
		P.x = -1;
		P.y = -1;
		return make_tuple(P, -1);
	}
	// eleminating edges
	double trace = Hessian.at<float>(0, 0) + Hessian.at<float>(1, 1);
	double deter = Hessian.at<float>(0, 0) * Hessian.at<float>(1, 1) - Hessian.at<float>(0, 1) * Hessian.at<float>(1, 0);
	double curvature = (trace * trace / deter);
	if (deter < 0 || curvature > 10)   // curv_threshold = 10
	{
		P.x = -1;
		P.y = -1;
		return make_tuple(P, -1);
	}
	P.x = jj * pow(2, octave_index);
	P.y = ii * pow(2, octave_index);

	return make_tuple(P, sc);
}

//-------------------------------------------------------------------------------

//-------------------------------------------------------------------------------

/*
	create scale space and difference of gaussian images
	inputs:
	- input image.
	- number of intervals.
	- vector of sigma values.
	outputs:
	- tuple of scale space and difference of gaussian images.
*/
tuple<vector<vector<Mat>>, vector<vector<Mat>>> Utility::Create_Scale_Space(Mat scr, int num_intervals, vector<double> sigmas)
{
	int num_octaves = compute_Number_Of_Octaves(scr.size());
	// define vector of vectors of mats for both scale space and DoG images.
	vector<vector<Mat>> scale_space;
	vector<vector<Mat>> DoG;
	// initialize both scale space and DoG
	for (int oct = 0; oct < num_octaves; oct++)
	{
		scale_space.push_back(vector<Mat>(num_intervals + 3));
		DoG.push_back(vector<Mat>(num_intervals + 2));
	}
	// blur the base image with first sigma value 
	GaussianBlur(scr, scr, Size(0, 0), sigmas[0], sigmas[0]);
	// copy the blurred image to the first octave and first scale image
	scr.copyTo(scale_space[0][0]);
	// two for loops for compute scale images and DoG images in all octaves
	for (int oct = 0; oct < num_octaves; oct++)
	{

		for (int scale = 1; scale < num_intervals + 3; scale++)
		{
			GaussianBlur(scale_space[oct][scale - 1], scale_space[oct][scale], Size(0, 0), sigmas[scale], sigmas[scale]);
			DoG[oct][scale - 1] = scale_space[oct][scale] - scale_space[oct][scale - 1];
		}

		// downsampling until reach the final octave
		if (oct < num_octaves - 1)
		{
			resize(scale_space[oct][0], scale_space[oct + 1][0], Size(0, 0), 0.5, 0.5, INTER_LINEAR);
		}

	}
	return make_tuple(scale_space, DoG);
}

//-------------------------------------------------------------------------------

/*
 	create list of extrema point
	inputs:
	- DOG images list.
	- scale space images list.
	- number of intervals.
	- sigma value.
	outputs:
	- vector of tuple of extrema point and it's scale.
*/
vector<tuple<Point, int>> Utility::Extrema(vector<vector<Mat>> DoG, vector<vector<Mat>> scale_images, int num_intervals, double sigma)
{
	tuple<Point, int> candidate;
	vector<tuple<Point, int>> result;
	double threshold = floor(0.5 * 0.04 / num_intervals * 255); // contrast_threshold=0.04
	vector<KeyPoint> keypoints;
	for (int oct = 0; oct < DoG.size(); oct++)
	{
		for (int scale = 1; scale < DoG[0].size() - 1; scale++)
		{
			for (int i = 5; i < DoG[oct][0].size().height - 5; i++)
			{
				for (int j = 5; j < DoG[oct][0].size().width - 5; j++)
				{
					// Check for the extrema value at present position
					// First argument is present DOG window at present position
					// Second argument is pos DOG window
					// Third argument is previous DOG window
					// Last argument is for thresholding
					if (is_it_extremum(DoG[oct][scale](Rect(j - 1, i - 1, 3, 3)),
						DoG[oct][scale + 1](Rect(j - 1, i - 1, 3, 3)),
						DoG[oct][scale - 1](Rect(j - 1, i - 1, 3, 3)), threshold))
					{
						// Estimate the actual location of present extrema along with it's scale 
						candidate = localize_Extremum(i, j, scale, DoG[oct], oct);

						// Check for available location candidate location != (-1,-1)
						if (get<0>(candidate).x != -1 && get<0>(candidate).y != -1)
						{
							/*vector<float> Orientations = compute_orientation(get<0>(candidate), oct, scale, scale_images[oct][get<1>(candidate)]);

							for (int angle = 0; angle < Orientations.size(); angle++)
							{
								KeyPoint key;
								key.angle = Orientations[angle];
								key.pt = get<0>(candidate);
								key.octave = oct;
								key.size = get<1>(candidate);
								keypoints.push_back(key);
							}*/

							result.push_back(candidate);

						}
					}

				}
			}
		}
	}
	//vector<KeyPoint> unique_keys;
	//unique_keys = remove_duplicate(keypoints);
	return result;
}

//-------------------------------------------------------------------------------

