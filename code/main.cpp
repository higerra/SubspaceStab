#include <iostream>
#include <gflags/gflags.h>
#include <opencv2/opencv.hpp>
#include <string>
#include "tracking.h"
#include "factorization.h"
#include "warping.h"
#include "thread_guard.h"

using namespace std;
using namespace cv;
using namespace Eigen;
using namespace substab;

DEFINE_int32(tWindow, 40, "tWindow");
DEFINE_int32(stride, 5, "Stride");
DEFINE_int32(kernelR, 10, "Radius of kernel");

void importVideo(const std::string& path, std::vector<cv::Mat>& images);

int main(int argc, char** argv){
	if(argc < 2){
		cerr << "Usage ./SubspaceStab <path-to-data>" << endl;
		return 1;
	}
	google::InitGoogleLogging(argv[0]);
	google::ParseCommandLineFlags(&argc, &argv, true);
	char buffer[1024] = {};
	vector<Mat> images;
	printf("Reading video...\n");
	importVideo(string(argv[1]), images);

	//tracking
	FeatureTracks trackMatrix;
	printf("Computing track matrix...\n");
	Tracking::genTrackMatrix(images, trackMatrix, FLAGS_tWindow, FLAGS_stride);
	printf("Done\n");


//	printf("Filtering tracks..\n");
//	int oriSize = (int)trackMatrix.offset.size();
//	Tracking::filterDynamicTracks(trackMatrix, (int)images.size());
//	int filteredSize = (int)trackMatrix.offset.size();
//	printf("%d/%d are kept\n", filteredSize, oriSize);
	//Tracking::visualizeTrack(images, trackMatrix, 10);

	Eigen::MatrixXd coe, bas, smoothedBas;
	vector<vector<int> > wMatrix(trackMatrix.offset.size());
	for(auto tid=0; tid<wMatrix.size(); ++tid)
		wMatrix[tid].resize(images.size(), 0);

	Factorization::movingFactorization(images, trackMatrix, coe, bas, wMatrix, FLAGS_tWindow, FLAGS_stride);

	Factorization::trackSmoothing(bas, smoothedBas, FLAGS_kernelR, -1);
	//warping
	GridWarpping warping(images[0].cols, images[0].rows);

	MatrixXd reconSmo = coe * smoothedBas;
	MatrixXd reconOri = coe * bas;

	CHECK_EQ(reconOri.rows(), trackMatrix.offset.size()*2);
	CHECK_EQ(reconOri.cols(), images.size());

	//reconstruction error
	double overallError = 0.0;
	double overallCount = 0.0;
	for(auto tid=0; tid<trackMatrix.offset.size(); ++tid){
		const int offset = (int)trackMatrix.offset[tid];
		for(auto v=offset; v<offset+trackMatrix.tracks[tid].size(); ++v){
			if(wMatrix[tid][v] != 1)
				continue;
			if(v >= images.size() - FLAGS_tWindow)
				continue;
			Vector2d oriPt(trackMatrix.tracks[tid][v-offset].x,trackMatrix.tracks[tid][v-offset].y);
			Vector2d reconPt = reconOri.block(2*tid, v, 2, 1);
			overallCount += 1.0;
			overallError += (reconPt-oriPt).norm();
		}
	}
	printf("Overall reconstruction error:%d/%d=%.3f\n", (int)overallError, (int)overallCount, overallError / overallCount);

	printf("Warping...\n");

	for(auto i=0; i<images.size(); ++i){
		for(auto y=0; y<images[i].rows; ++y){
			for(auto x=0; x<images[i].cols; ++x){
				if(images[i].at<Vec3b>(y,x) == Vec3b(0,0,0))
					images[i].at<Vec3b>(y,x) = Vec3b(1,1,1);
			}
		}
	}

	vector<Mat> warped(images.size()-FLAGS_tWindow);
	const int num_thread = 7;
	vector<thread_guard> threads((size_t) num_thread);
	auto threadFunWarp = [&](int threadId) {
		for (auto v = threadId; v < images.size() - FLAGS_tWindow; v += num_thread) {
			vector<Vector2d> pts1, pts2;
			for (auto tid = 0; tid < trackMatrix.offset.size(); ++tid) {
				const int offset = (int) trackMatrix.offset[tid];
				if(wMatrix[tid][v] != 1)
					continue;
				if (trackMatrix.offset[tid] <= v && offset + trackMatrix.tracks.size() >= v && wMatrix[tid][v]) {
					pts1.push_back(Vector2d(reconOri(2 * tid, v), reconOri(2 * tid + 1, v)));
					pts2.push_back(Vector2d(reconSmo(2 * tid, v), reconSmo(2 * tid + 1, v)));
//					printf("(%.3f,%.3f), (%.3f,%.3f)\n", pts1.back()[0], pts1.back()[1], pts2.back()[0], pts2.back()[1]);
				}
			}
			printf("Frame %d on thread %d\n", v, threadId);
			printf("number of constraints: %d\n", (int) pts1.size());
			warping.warpImage(images[v], warped[v], pts1, pts2);
			for(auto ftid=0; ftid<pts2.size(); ++ftid)
				cv::circle(warped[v],cv::Point2d(pts2[ftid][0], pts2[ftid][1]),1,Scalar(0,0,255),2);
		}
	};

	for(auto tid=0; tid<threads.size(); ++tid){
		std::thread t(threadFunWarp, tid);
		threads[tid].bind(t);
	}

	for(auto& t: threads)
		t.join();


//	//crop
	int top=0, bottom=warped[0].rows-1, left=0, right=warped[0].cols-1;
	for(;top<warped[0].rows; ++top){
		bool is_border = false;
		for(auto v=0; v<warped.size(); ++v){
			for(auto x=0; x<warped[v].cols; ++x) {
				Vec3b pix = warped[v].at<Vec3b>(top, x);
				if(pix == Vec3b(0,0,0)){
					is_border = true;
					break;
				}
			}
			if(is_border)
				break;
		}
		if(!is_border)
			break;
	}
	for(; bottom>=0; --bottom){
		bool is_border = false;
		for(auto v=0; v<warped.size(); ++v){
			for(auto x=0; x<warped[v].cols; ++x) {
				Vec3b pix = warped[v].at<Vec3b>(bottom, x);
				if(pix == Vec3b(0,0,0)){
					is_border = true;
					break;
				}
			}
			if(is_border)
				break;
		}
		if(!is_border)
			break;
	}
	for(; right>=0; --right){
		bool is_border = false;
		for(auto v=0; v<warped.size(); ++v){
			for(auto y=0; y<warped[v].rows; ++y) {
				Vec3b pix = warped[v].at<Vec3b>(y, right);
				if(pix == Vec3b(0,0,0)){
					is_border = true;
					break;
				}
			}
			if(is_border)
				break;
		}
		if(!is_border)
			break;
	}
	for(; left<warped[0].cols; ++left){
		bool is_border = false;
		for(auto v=0; v<warped.size(); ++v){
			for(auto y=0; y<warped[v].rows; ++y) {
				Vec3b pix = warped[v].at<Vec3b>(y, left);
				if(pix == Vec3b(0,0,0)){
					is_border = true;
					break;
				}
			}
			if(is_border)
				break;
		}
		if(!is_border)
			break;
	}
	printf("Range:(%d,%d,%d,%d)\n", left, right, top, bottom);
	for(auto v=0; v<warped.size(); ++v){
		//Mat out = warped[v].colRange(left,right).rowRange(top,bottom);
		sprintf(buffer, "warped%05d.jpg", v);
		imwrite(buffer, warped[v]);
	}


	{
		//debug:
//		FeatureTracks trackMatrix2;
//		trackMatrix2.offset = trackMatrix.offset;
//		trackMatrix2.tracks.resize(trackMatrix2.offset.size());
//		for(auto tid=0; tid < trackMatrix2.offset.size(); ++tid){
//			for(auto v=trackMatrix2.offset[tid]; v<trackMatrix2.offset[tid] + trackMatrix.tracks[tid].size(); ++v){
//				const double x = recon(2*tid, v);
//				const double y = recon(2*tid+1, v);
//				trackMatrix2.tracks[tid].push_back(cv::Point2f(x,y));
//			}
//		}
//		Tracking::visualizeTrack(images, trackMatrix2, 10);

	}


    return 0;
}

void importVideo(const std::string& path, std::vector<cv::Mat>& images){
	VideoCapture cap(path);
	const cv::Size dsize(640,360);
	CHECK(cap.isOpened()) << "Can not open video " << path;
	while(true){
		Mat frame;
		bool success = cap.read(frame);
		if(!success)
			break;
		cv::resize(frame, frame, dsize);
		images.push_back(frame);
	}
}