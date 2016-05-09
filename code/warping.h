//
// Created by yanhang on 4/21/16.
//

#ifndef SUBSPACESTAB_WARPPING_H
#define SUBSPACESTAB_WARPPING_H

#include <opencv2/opencv.hpp>
#include <Eigen/Eigen>
#include <iostream>
#include <glog/logging.h>

namespace substab {

	class GridWarpping {
	public:
		GridWarpping(const int w, const int h, const int gw = 64, const int gh = 36);


		void getGridIndAndWeight(const Eigen::Vector2d &pt, Eigen::Vector4i &ind, Eigen::Vector4d &w) const;

		//wf: output dense warping field
		void warpImage(const cv::Mat& input, cv::Mat& output, const std::vector<Eigen::Vector2d>& pts1, const std::vector<Eigen::Vector2d>& pts2)const;
		//void warpImageCloseForm(const cv::Mat& input, cv::Mat& output, const std::vector<Eigen::Vector2d>& pts1, const std::vector<Eigen::Vector2d>& pts2) const;

		void computeSimilarityWeight(const cv::Mat& input, std::vector<Eigen::Vector2d>& saliency) const;

		inline double getBlockW() const{
			return blockW;
		}
		inline double getBlockH() const{
			return blockH;
		}

		void visualizeGrid(const std::vector<Eigen::Vector2d>& grid, cv::Mat& img) const;
	private:
		std::vector<Eigen::Vector2d> gridLoc;

		int width;
		int height;
		int gridW;
		int gridH;
		double blockW;
		double blockH;
	};

}//namespace substab

#endif //SUBSPACESTAB_WARPPING_H
