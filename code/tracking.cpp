//
// Created by yanhang on 4/21/16.
//

#include "tracking.h"

using namespace std;
using namespace cv;

namespace substab{
	namespace Tracking{
		void genTrackMatrix(const std::vector<cv::Mat>& images, FeatureTracks& trackMatrix, const int tWindow, const int stride){
			CHECK(!images.empty());
			const int width = images[0].cols;
			const int height = images[0].rows;
			char buffer[1024] = {};

			printf("width: %d, height:%d\n", width, height);

			int N = 0;
			vector<Mat> grays(images.size());
			for(auto v=0; v<grays.size(); ++v)
				cvtColor(images[v], grays[v], CV_BGR2GRAY);

			const double quality_level = 0.02;
			const double min_distance = 20;
			const int winSizePyramid = 21;
			const int nLevel = 2;

			const double max_diff_distance = 2;
			const int max_corners = width * height / 64;
			const int interval = 1;

			printf("Building image pyramid...\n");
			vector<vector<cv::Mat> > pyramid(images.size());
			for(auto v=0; v<images.size(); ++v) {
				cv::buildOpticalFlowPyramid(grays[v], pyramid[v], cv::Size(winSizePyramid, winSizePyramid), nLevel);
			}

			for(auto v=0; v<images.size() - tWindow - stride; v+=interval){
				CHECK_EQ(trackMatrix.tracks.size(), trackMatrix.offset.size());
				printf("==========================\n");
				printf("Start frame: %d\n", v);
				vector<cv::Point2f> corners;
				cv::goodFeaturesToTrack(grays[v], corners, max_corners, quality_level, min_distance);
				vector<cv::Point2f> newcorners;
				newcorners.reserve(corners.size());
				for(auto cid=0; cid < corners.size(); ++cid){
					bool is_new = true;
					for(auto j=0; j<trackMatrix.tracks.size(); ++j){
						if(trackMatrix.tracks[j].size() + trackMatrix.offset[j] >= v){
							const cv::Point2f& pt = trackMatrix.tracks[j][v-trackMatrix.offset[j]];
							double dis = cv::norm(corners[cid]-pt);
							if (dis <= max_diff_distance)
								is_new = false;
						}
					}
					if(is_new)
						newcorners.push_back(corners[cid]);
				}

				vector<vector<cv::Point2f> > curTracks(newcorners.size());
				for(auto i=0; i<curTracks.size(); ++i)
					curTracks[i].push_back(newcorners[i]);

				vector<bool> is_valid(newcorners.size(), true);
				for(auto i=v+1; i<images.size(); ++i){
					if(newcorners.empty())
						break;
					vector<cv::Point2f> tracked;
					vector<uchar> status;
					Mat err;
					cv::calcOpticalFlowPyrLK(pyramid[i-1], pyramid[i], newcorners, tracked, status, err, cv::Size(winSizePyramid, winSizePyramid));
					CHECK_EQ(tracked.size(), curTracks.size());
					for(auto j=0; j<status.size(); ++j){
						if(status[j] == (uchar)1 && is_valid[j]) {
							curTracks[j].push_back(tracked[j]);
						} else
							is_valid[j] = false;
					}
					newcorners.swap(tracked);
				}

				for(auto i=0; i<curTracks.size(); ++i){
					if(curTracks[i].size() < tWindow)
						continue;
					trackMatrix.tracks.push_back(curTracks[i]);
					trackMatrix.offset.push_back((size_t)v);
				}
			}

		}

		void visualizeTrack(const std::vector<cv::Mat>& images, const FeatureTracks& trackMatrix, const int startFrame) {
			const int testStartFrame = startFrame;
			char buffer[1024] = {};
			//collect trackes to visualize
			for (auto v = testStartFrame; v < images.size(); ++v) {
				//if no tracks, break
				int kTrack = 0;
				Mat img = images[v].clone();
				for (auto tid = 0; tid < trackMatrix.tracks.size(); ++tid) {
					if (trackMatrix.offset[tid] == testStartFrame &&
					    trackMatrix.tracks[tid].size() + trackMatrix.offset[tid] >= v) {
						cv::circle(img, trackMatrix.tracks[tid][v - trackMatrix.offset[tid]], 1, Scalar(0, 0, 255), 2);
						kTrack++;
					}
				}
				if (kTrack == 0)
					break;
				sprintf(buffer, "trackVisOri%05d_t%05d.jpg", testStartFrame, v);
				imwrite(buffer, img);
			}
		}

		void filterDynamicTracks(FeatureTracks& trackMatrix, const int N){
			const int stride = 5;
			const double max_error = 2;

			vector<bool> keep(trackMatrix.tracks.size(), true);

			for(auto v=0; v<N-stride; ++v){
				vector<cv::Point2f> pts1, pts2;
				vector<size_t> trackId;
				for(auto tid=0; tid<trackMatrix.tracks.size(); ++tid){
					if(trackMatrix.offset[tid] <= v && trackMatrix.offset[tid]+trackMatrix.tracks[tid].size() >= v+stride){
						pts1.push_back(trackMatrix.tracks[tid][v-trackMatrix.offset[tid]]);
						pts2.push_back(trackMatrix.tracks[tid][v+stride-trackMatrix.offset[tid]]);
					}
				}
			}
		}
	}//namespace Tracking
}//namespace substab