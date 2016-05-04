#include <iostream>
#include <gflags/gflags.h>
#include <opencv2/opencv.hpp>
#include <string>
#include "tracking.h"
#include "factorization.h"
#include "warping.h"

using namespace std;
using namespace cv;
using namespace Eigen;
using namespace substab;

DEFINE_int32(tWindow, 30, "tWindow");
DEFINE_int32(stride, 5, "Stride");

void importVideo(const std::string& path, std::vector<cv::Mat>& images);

int main(int argc, char** argv){
	if(argc < 2){
		cerr << "Usage ./SubspaceStab <path-to-data>" << endl;
		return 1;
	}
	google::InitGoogleLogging(argv[0]);
	char buffer[1024] = {};
	vector<Mat> images;
	printf("Reading video...\n");
	importVideo(string(argv[1]), images);

	//tracking
	FeatureTracks trackMatrix;
	printf("Computing track matrix...\n");
	Tracking::genTrackMatrix(images, trackMatrix, FLAGS_tWindow, FLAGS_stride);
	printf("Done\n");

	Mat coe, bas;
	movingFactorization(trackMatrix, coe, bas, (int)images.size(), FLAGS_tWindow, FLAGS_stride);
//	printf("%d frames read\n", (int)images.size());
//	for(auto i=0; i<images.size(); ++i){
//		sprintf(buffer, "frame%05d.jpg", i);
//		imwrite(buffer, images[i]);
//	}
    return 0;
}

void importVideo(const std::string& path, std::vector<cv::Mat>& images){
	VideoCapture cap(path);
	CHECK(cap.isOpened()) << "Can not open video " << path;
	while(true){
		Mat frame;
		bool success = cap.read(frame);
		if(!success)
			break;
		images.push_back(frame);
	}
}