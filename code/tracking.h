//
// Created by yanhang on 4/21/16.
//

#ifndef SUBSPACESTAB_TRACKING_H
#define SUBSPACESTAB_TRACKING_H

#include <opencv2/opencv.hpp>
#include <Eigen/Eigen>
#include <vector>
#include <string>
#include <glog/logging.h>

namespace substab {
	struct FeatureTracks{
		std::vector<std::vector<cv::Point2f> > tracks;
		std::vector<size_t> offset;
	};

	namespace Tracking {
		void genTrackMatrix(const std::vector<cv::Mat>& images, FeatureTracks& trackMatrix, const int tWindow, const int stride);
		void filterDynamicTracks(FeatureTracks& trackMatrix, const int N);

		void visualizeTrack(const std::vector<cv::Mat>& images, const FeatureTracks& trackMatrix, const int startFrame);
	}
}
#endif //SUBSPACESTAB_TRACKING_H
