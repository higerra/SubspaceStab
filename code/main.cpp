#include <iostream>
#include <gflags/gflags.h>
#include <opencv2/opencv.hpp>
#include <string>
#include "tracking.h"
#include "warping.h"

using namespace std;
using namespace cv;
using namespace Eigen;
using namespace substab;

void importVideo(const std::string& path, std::vector<cv::Mat>& images);

int main(int argc, char** argv){
	if(argc < 2){
		cerr << "Usage ./SubspaceStab <path-to-data>" << endl;
		return 1;
	}

	char buffer[1024] = {};
	vector<Mat> images;
	importVideo(string(argv[1]), images);

	//tracking
	MatrixXf tracks;
	printf("Computing track matrix...\n");
	Tracking::genTrackMatrix(images, tracks);
	printf("Done\n");

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
	cout << cap.get(CV_CAP_PROP_FRAME_COUNT) << endl;

	while(true){
		Mat frame;
		bool success = cap.read(frame);
		if(!success)
			break;
		images.push_back(frame);
	}
}