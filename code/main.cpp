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
DEFINE_int32(kernelR, -1, "Radius of kernel");
DEFINE_int32(num_thread, 6, "number of threads");
DEFINE_string(output, "", "output file name");
DEFINE_bool(crop, true, "crop the output video");
DEFINE_bool(draw_points, false, "draw feature points");
DEFINE_bool(resize, true, "resize to 640 * 360");

void importVideo(const std::string& path, std::vector<cv::Mat>& images, double& fps, int& vcodec);
void cropImage(const std::vector<cv::Mat>& input, std::vector<cv::Mat>& output);

int main(int argc, char** argv){
	if(argc < 2){
		cerr << "Usage ./SubspaceStab <path-to-input-video>" << endl;
		return 1;
	}
	google::InitGoogleLogging(argv[0]);
	google::ParseCommandLineFlags(&argc, &argv, true);

	char buffer[1024] = {};
	vector<Mat> images;
	// printf("Reading in video file ... \n");
	int vcodec; double frameRate;
   	importVideo(string(argv[1]), images, frameRate, vcodec);

	//tracking features
	FeatureTracks trackMatrix;
	// printf("Computing track matrix...\n");
	Tracking::genTrackMatrix(images, trackMatrix, FLAGS_tWindow, FLAGS_stride);
	// printf("Done\n");
	Eigen::MatrixXd coe, bas, smoothedBas;

	vector<vector<int> > wMatrix(trackMatrix.offset.size());
	for(auto tid=0; tid<wMatrix.size(); ++tid)
		wMatrix[tid].resize(images.size(), 0);

	// printf("Factorization...\n");
	Factorization::movingFactorization(images, trackMatrix, coe, bas, wMatrix, FLAGS_tWindow, FLAGS_stride);
	const int smoothR = FLAGS_kernelR > 0 ? FLAGS_kernelR : FLAGS_tWindow / 2;

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
	printf("Overall reconstruction error:%.3f\n", overallError / overallCount);

	printf("Smoothing...\n");
	Factorization::trackSmoothing(bas, smoothedBas, smoothR, -1);
	MatrixXd reconSmo = coe * smoothedBas;
	CHECK_EQ(reconSmo.rows(), trackMatrix.offset.size()*2);
	CHECK_EQ(reconSmo.cols(), images.size());

	printf("Warping...\n");
	GridWarpping warping(images[0].cols, images[0].rows);
	for(auto i=0; i<images.size(); ++i){
		for(auto y=0; y<images[i].rows; ++y){
			for(auto x=0; x<images[i].cols; ++x){
				if(images[i].at<Vec3b>(y,x) == Vec3b(0,0,0))
					images[i].at<Vec3b>(y,x) = Vec3b(1,1,1);
			}
		}
	}
	vector<Mat> warped(images.size()-FLAGS_tWindow);

	const int num_thread = FLAGS_num_thread;

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
				}
			}
//			printf("Frame %d on thread %d\n", v, threadId);
//			printf("number of constraints: %d\n", (int) pts1.size());
			warping.warpImageCloseForm(images[v], warped[v], pts1, pts2,v);
			if(FLAGS_draw_points) {
				for (auto ftid = 0; ftid < pts2.size(); ++ftid)
					cv::circle(warped[v], cv::Point2d(pts2[ftid][0], pts2[ftid][1]), 1, Scalar(0, 0, 255), 2);
			}
		}
	};

	for(auto tid=0; tid<threads.size(); ++tid){
		std::thread t(threadFunWarp, tid);
		threads[tid].bind(t);
	}

	for(auto& t: threads)
		t.join();
	printf("Done\n");

	if(FLAGS_crop) {
		printf("Cropping...\n");
		vector<Mat> croped;
		cropImage(warped, croped);
		warped.swap(croped);
		printf("Done\n");
	}
	CHECK(!warped.empty());

	string outputFileName = FLAGS_output;
	if(FLAGS_output.empty()){
		printf("Output file name not provided. Write to 'stabilized.mp4'\n");
		outputFileName = "stabilized";
	}

	for(auto v=0; v<warped.size(); ++v){
//		Mat out = warped[v].colRange(left,right).rowRange(top,bottom);
		sprintf(buffer, "%s%05d.jpg", outputFileName.c_str(), v);
		imwrite(buffer, warped[v]);
//		vwriter.write(warped[v]);
	}
    return 0;
}

void importVideo(const std::string& path, std::vector<cv::Mat>& images, double& fps, int& vcodec){
	VideoCapture cap(path);
	CHECK(cap.isOpened()) << "Can not open video " << path;
	cv::Size dsize(640,320);
	while(true){
		Mat frame;
		bool success = cap.read(frame);
		if(!success)
			break;
		if(FLAGS_resize)
			cv::resize(frame, frame, dsize);
		images.push_back(frame);
	}
	fps = cap.get(cv::CAP_PROP_FPS);
	vcodec = (int)cap.get(cv::CAP_PROP_FOURCC);

}


void cropImage(const std::vector<cv::Mat>& input, std::vector<cv::Mat>& output){
	CHECK(!input.empty());
	const int imgWidth = input[0].cols;
	const int imgHeight = input[0].rows;
	Mat cropMask(imgHeight, imgWidth, CV_32F, Scalar::all(0));
	for(auto y=0; y<imgHeight; ++y){
		for(auto x=0; x<imgWidth; ++x){
			bool has_black = false;
			for(auto v=0; v<input.size(); ++v){
				if(input[v].at<Vec3b>(y,x) == Vec3b(0,0,0)){
					has_black = true;
					break;
				}
			}
			if(has_black)
				cropMask.at<float>(y,x) = -1000;
			else
				cropMask.at<float>(y,x) = 1;
		}
	}
	Mat integralImage;
	cv::integral(cropMask, integralImage, CV_32F);
	Vector4i roi;
	float optValue = -1000 * imgWidth * imgHeight;
	const int stride = 3;
	for(auto x1=0; x1<imgWidth; x1+=stride) {
		for (auto y1 = 0; y1 < imgHeight; y1+=stride) {
			for (auto x2 = x1 + stride; x2 < imgWidth; x2+=stride) {
				for (auto y2 = y1 + stride; y2 < imgHeight; y2+=stride) {
					float curValue = integralImage.at<float>(y2, x2) + integralImage.at<float>(y1, x1)
									 - integralImage.at<float>(y2, x1) - integralImage.at<float>(y1, x2);
					if(curValue > optValue){
						optValue = curValue;
						roi = Vector4i(x1,y1,x2,y2);
					}
				}
			}
		}
	}
	printf("roi:%d,%d,%d,%d\n", roi[0], roi[1], roi[2], roi[3]);

	output.resize(input.size());
	for(auto i=0; i<output.size(); ++i){
		output[i] = input[i].colRange(roi[0],roi[2]).rowRange(roi[1], roi[3]).clone();
		cv::resize(output[i], output[i], cv::Size(imgWidth, imgHeight));
	}
}
