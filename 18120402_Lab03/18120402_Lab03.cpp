#include "opencv2/opencv.hpp "
#include "opencv2/highgui/highgui.hpp"
#include <string>
#include "Point_Detector.h"
#include "SIFT.h"

using namespace cv;
using namespace std;


//int main(int argc, char** argv)
//{
//	// Check number of argument is correct
//	if (argc < 2 && argc > 3)
//	{
//		cout << "Photoshop program" << endl;
//		return -1;
//	}
//
//	// Load color image
//	Mat src_img = imread(argv[1], IMREAD_COLOR);
//	if (!src_img.data)
//	{
//		cout << "Cannot open image" << std::endl;
//		return -1;
//	}
//
//	// Intialization
//	Mat dst_img;
//	int code = atoi(argv[2]);
//	//int optional = atoi(argv[3]);
//	Point_Detector pd;
//	int result = 1;
//
//	/* Point Detection using Harris's Algorithm */
//	if (code == 0)
//	{
//		cout << code;
//		dst_img = pd.detectHarrist(src_img);
//	}
//
//	/* Point Detection using Blob's Algorithm */
//	else if (code == 1)
//	{
//		cout << code;
//		dst_img = pd.detectBlob(src_img);
//	}
//
//	/* Point Detection using DOG Algorithm */
//	else if (code == 2)
//	{
//		cout << code;
//		dst_img = pd.detectDOG(src_img);
//	}
//
//	/* Matching 2 image using KNN algorithm */
//	else if (code == 2)
//	{
//		cout << code;
//		//dst_img = pd.matchBySIFT(src_img1, src_img2, method);
//	}
//
//	if (result == 1)
//	{
//		imshow("Source Image", src_img);
//		imshow("Destination Image", dst_img);
//
//		waitKey(0);
//	}
//	else
//	{
//		cout << "Cannot open image\n";
//	}
//
//	return 0;
//}

int main()
{
	// Load color image
	string input_file = "C:\\Users\\pc\\Desktop\\Temparory\\room.jpg";
	Mat src_img = imread(input_file, IMREAD_COLOR);
	if (!src_img.data)
	{
		cout << "Cannot open image" << std::endl;
		return -1;
	}

	// Intialization
	Mat dst_img;
	int code = 3;
	//int optio;al = atoi(argv[3]);
	Point_Detector pd;
	int result = 1;

	/* Point Detection using Harris's Algorithm */
	if (code == 0)
	{
		cout << code;
		dst_img = pd.detectHarrist(src_img);
	}

	/* Point Detection using Blob's Algorithm */
	else if (code == 1)
	{
		cout << code;
		dst_img = pd.detectBlob(src_img);
	}

	/* Point Detection using DOG Algorithm */
	else if (code == 2)
	{
		cout << code;
		dst_img = pd.detectDOG(src_img);
	}

	/* Matching 2 image using KNN algorithm */
	else if (code == 3)
	{
		cout << code;
		Sift_feature Sift;
		dst_img = Sift.detect(src_img);
	}

	if (result == 1)
	{
		imshow("Source Image", src_img);
		imshow("Destination Image", dst_img);

		waitKey(0);
	}
	else
	{
		cout << "Cannot open image\n";
	}

	return 0;
}

