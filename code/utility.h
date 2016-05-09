//
// Created by Yan Hang on 1/28/16.
//

#ifndef RENDERPROJECT_INTERPOLATION_H
#define RENDERPROJECT_INTERPOLATION_H

#include <Eigen/Eigen>
#include <iostream>
#include <algorithm>
#include <numeric>
#include <glog/logging.h>

namespace interpolation_util {
	template<typename T, int N>
	Eigen::Matrix<double, N, 1> bilinear(const T *const data, const int w, const int h, const Eigen::Vector2d &loc) {
		using namespace Eigen;
		const double epsilon = 0.00001;
		int xl = floor(loc[0] - epsilon), xh = (int) round(loc[0] + 0.5 - epsilon);
		int yl = floor(loc[1] - epsilon), yh = (int) round(loc[1] + 0.5 - epsilon);

		if (loc[0] <= epsilon)
			xl = 0;
		if (loc[1] <= epsilon)
			yl = 0;

		const int l1 = yl * w + xl;
		const int l2 = yh * w + xh;
		if (l1 == l2) {
			Matrix<double, N, 1> res;
			for (size_t i = 0; i < N; ++i)
				res[i] = data[l1 * N + i];
			return res;
		}

//	char buffer[100] = {};
//		sprintf(buffer, "bilinear(): coordinate out of range: (%.2f,%.2f), (%d,%d,%d,%d), l1:%d, l2:%d!",
//		        loc[0], loc[1], xl, yl, xh, yh, l1, l2);
		CHECK(!(l1 < 0 || l2 < 0 || l1 >= w * h || l2 >= w * h)) << loc[0] << ' ' << loc[1] << ' '<< w << ' '<< h;

		double lm = loc[0] - (double) xl, rm = (double) xh - loc[0];
		double tm = loc[1] - (double) yl, bm = (double) yh - loc[1];
		Vector4i ind(xl + yl * w, xh + yl * w, xh + yh * w, xl + yh * w);

		std::vector<Matrix<double, N, 1>> v(4);
		for (size_t i = 0; i < 4; ++i) {
			for (size_t j = 0; j < N; ++j)
				v[i][j] = data[ind[i] * N + j];
		}
		if (std::abs(lm) <= epsilon && std::abs(rm) <= epsilon)
			return (v[0] * bm + v[2] * tm) / (bm + tm);

		if (std::abs(bm) <= epsilon && std::abs(tm) <= epsilon)
			return (v[0] * rm + v[2] * lm) / (lm + rm);

		Vector4d vw(rm * bm, lm * bm, lm * tm, rm * tm);
		double sum = vw.sum();
//	sprintf(buffer, "loc:(%.2f,%.2f), integer: (%d,%d,%d,%d), margin: (%.2f,%.2f,%.2f,%.2f), sum: %.2f",
//		loc[0], loc[1], xl, yl, xh, yh, lm, rm, tm, bm, sum);
		CHECK_GT(sum, 0);
		return (v[0] * vw[0] + v[1] * vw[1] + v[2] * vw[2] + v[3] * vw[3]) / sum;
	};

	template<typename T, int N>
	Eigen::Matrix<double, N, 1> nearest(const T *const data, const int w, const int h, const Eigen::Vector2d &loc) {
		using namespace Eigen;
		int xl = floor(loc[0]), xh = (int) round(loc[0] + 0.5);
		int yl = floor(loc[1]), yh = (int) round(loc[1] + 0.5);
		const int l1 = yl * w + xl;
		const int l2 = yh * w + xh;
		Vector4i ind(xl + yl * w, xh + yl * w, xh + yh * w, xl + yh * w);
		std::vector<Vector2d> coord(4);
		coord[0] = Vector2d(xl, yl);
		coord[1] = Vector2d(xh, yl);
		coord[2] = Vector2d(xh, yh);
		coord[3] = Vector2d(xl, yh);
		Matrix<double, N, 1> res;
		double min_dis = 9999999;
		for (size_t i = 0; i < 4; ++i) {
			if (ind[i] < 0 || ind[i] >= w * h)
				continue;
			double curdis = (loc[0] - coord[i][0]) * (loc[0] - coord[i][0]) +
							(loc[1] - coord[i][1]) * (loc[1] - coord[i][1]);
			if (curdis < min_dis) {
				min_dis = curdis;
				for (size_t j = 0; j < N; ++j)
					res[j] = data[ind[i] * N + j];
			}
		}
		CHECK_LT(min_dis, 1.0);
		return res;
	};


#ifdef USE_CUDA

#endif
}//namespace interpolate_util

namespace imgproc_util{

}

namespace math_util {
	inline double variance(const std::vector<double> &a, const double mean) {
		CHECK_GT(a.size(),1);
		const double n = (double) a.size();
		std::vector<double> diff(a.size());
		std::transform(a.begin(), a.end(), diff.begin(), std::bind2nd(std::minus<double>(), mean));
		return std::sqrt(std::inner_product(diff.begin(), diff.end(), diff.begin(), 0.0) / (n-1));
	}

	inline double variance(const std::vector<double> &a) {
		CHECK(!a.empty());
		const double mean = std::accumulate(a.begin(), a.end(), 0.0) / (double) a.size();
		return variance(a, mean);
	}

	double normalizedCrossCorrelation(const std::vector<double> &a1, const std::vector<double> &a2);

	inline double gaussian(const double m, const double sigma, const double v){
		CHECK_GT(sigma, 0);
		return std::exp(-1 * (v-m)*(v-m) / (2 * sigma * sigma));
	}

	inline double huberNorm(const double v, const double alpha){
		if(std::abs(v) < alpha)
			return 0.5 * v * v;
		else
			return alpha * (std::abs(v) - 0.5*alpha);
	}



}//namespace math_util

namespace geometry_util{
	template<int N>
	double distanceToLineSegment(const Eigen::Matrix<double,N,1>& pt, const Eigen::Matrix<double, N,1>& spt, const Eigen::Matrix<double, N,1>& ept){
		const double epsilon = 1e-6;
		typedef Eigen::Matrix<double, N,1> VecN;
		VecN sp = pt - spt;
		VecN lineDir = ept - spt;
		double lineNorm = lineDir.norm();
		CHECK_GT(lineNorm, epsilon);
		lineDir /= lineNorm;
		//p2: projection of pt to line
		VecN p2 = spt + lineDir * sp.dot(lineDir);
		//if p2 is outside line segment
		if((p2-spt).dot(ept-p2) < 0)
			return std::min((pt-spt).norm(), (pt-ept).norm());
		return (pt-p2).norm();
	}
}//namespace geometry_util


#endif //RENDERPROJECT_INTERPOLATION_H
