//
// Created by yanhang on 4/21/16.
//

#include "tracking.h"

using namespace std;
using namespace cv;

namespace substab{
	namespace Tracking{
		void genTrackMatrix(const std::vector<cv::Mat>& images, Eigen::MatrixXf& trackMatrix){
			CHECK(!images.empty());
			const int width = images[0].cols;
			const int height = images[0].rows;

			int N = 0;
			vector<vector<cv::Point2f> > tracks(images.size());

			for(auto& v: tracks)
				v.resize((size_t)width * height, cv::Point2f(-1,-1));

			const double quality_level = 0.02;
			const double min_distance = 20;
			const int winSizePyramid = 21;
			const int nLevel = 2;
			const size_t min_track_length = 10;
			const double max_diff_distance = 2;

			vector<vector<cv::Mat> > pyramid(images.size());
			for(auto v=0; v<images.size(); ++v)
				cv::buildOpticalFlowPyramid(images[v], pyramid[v], cv::Size(winSizePyramid, winSizePyramid), nLevel);


			int trackInd = 0;
			for(auto v=0; v<images.size(); ++v){
				vector<cv::Point2f> corners;
				cv::goodFeaturesToTrack(images[v], corners, width * height, quality_level, min_distance);
				vector<cv::Point2f> newcorners;
				newcorners.reserve(corners.size());
				for(auto cid=0; cid < corners.size(); ++cid){
					bool is_new = true;
					for(auto j=0; j<tracks[v].size(); ++j){
						double dis = cv::norm(corners[cid]-tracks[v][j]);
						if(dis <= max_diff_distance)
							is_new = false;
					}
					if(is_new)
						newcorners.push_back(corners[cid]);
				}

				vector<vector<cv::Point2f> > curTracks(newcorners.size());
				for(auto i=0; i<curTracks.size(); ++i)
					curTracks[i].push_back(newcorners[i]);

				vector<bool> is_valid(newcorners.size());
				for(auto i=v+1; v<images.size(); ++i){
					vector<cv::Point2f> tracked;
					vector<uchar> status;
					Mat err;
					cv::calcOpticalFlowPyrLK(pyramid[i-1], pyramid[i], newcorners, tracked, status, err, cv::Size(winSizePyramid, winSizePyramid));
					CHECK_EQ(tracked.size(), curTracks.size());
					for(auto j=0; j<status.size(); ++j){
						if(status[j] == (uchar)1 && is_valid[j])
							curTracks[j].push_back(tracked[j]);
						else
							is_valid[j] = false;
					}
					newcorners.swap(tracked);
				}
				bool increase_ind = false;
				for(auto i=0; i<curTracks.size(); ++i){
					if(curTracks[i].size() < min_track_length)
						continue;
					for(auto j=v; j<v+curTracks.size(); ++j)
						tracks[j][trackInd] = curTracks[i][j];
					increase_ind = true;
				}
				if(increase_ind) {
					trackInd++;
					CHECK_LT(trackInd, tracks[0].size());
				}
			}
		}
	}
}//namespace substab