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
	namespace Tracking {
		void genTrackMatrix(const std::vector<cv::Mat>& images, Eigen::MatrixXf& trackMatrix);
	}
}
#endif //SUBSPACESTAB_TRACKING_H
