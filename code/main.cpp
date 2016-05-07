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

	Eigen::MatrixXd coe, bas;
	movingFactorization(images, trackMatrix, coe, bas, images.size(), FLAGS_tWindow, FLAGS_stride);

	{
		//debug:
//		Mat recon = coe * bas;
//		FeatureTracks trackMatrix2;
//		trackMatrix2.offset = trackMatrix.offset;
//		trackMatrix2.tracks.resize(trackMatrix2.offset.size());
//		for(auto tid=0; tid < trackMatrix2.offset.size(); ++tid){
//			for(auto v=trackMatrix2.offset[tid]; v<trackMatrix2.offset[tid] + trackMatrix.tracks[tid].size(); ++v){
//				const double x = recon.at<double>(2*tid, v);
//				const double y = recon.at<double>(2*tid+1, v);
//				trackMatrix2.tracks[tid].push_back(cv::Point2f(x,y));
//			}
//		}
//		Tracking::visualizeTrack(images, trackMatrix2, 0);
//		Tracking::visualizeTrack(images, trackMatrix, 0);
	}


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