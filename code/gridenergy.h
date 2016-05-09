//
// Created by yanhang on 3/28/16.
//

#ifndef DYNAMICSTEREO_GRIDENERGY_H
#define DYNAMICSTEREO_GRIDENERGY_H
#include <ceres/ceres.h>
#include <Eigen/Eigen>

namespace substab{

	struct WarpFunctorData{
	public:
		WarpFunctorData(const Eigen::Vector2d& tgt_, const Eigen::Vector4d& w_, const double weight_): tgt(tgt_), w(w_), weight(std::sqrt(weight_)){}
		template<typename T>
		bool operator()(const T* const g1, const T* const g2, const T* const g3, const T* const g4, T* residual) const{
			T x = g1[0] * w[0] + g2[0] * w[1] + g3[0] * w[2] + g4[0] * w[3];
			T y = g1[1] * w[0] + g2[1] * w[1] + g3[1] * w[2] + g4[1] * w[3];
			T diffx = x - (T)tgt[0];
			T diffy = y - (T)tgt[1];
			residual[0] = weight * ceres::sqrt(diffx * diffx + diffy * diffy);
			return true;
		}
	private:
		const Eigen::Vector2d tgt;
		const Eigen::Vector4d w;
		const double weight;
	};

    struct WarpFunctorRegularization{
    public:
        WarpFunctorRegularization(const Eigen::Vector2d& pt_, const double w_): pt(pt_), w(std::sqrt(w_)){}
        template<typename T>
        bool operator()(const T* const g1, T* residual) const{
            T diffx = g1[0] - (T)pt[0];
            T diffy = g1[1] - (T)pt[1];
            residual[0] = ceres::sqrt((diffx * diffx + diffy * diffy) + 1e-10) * w;
            return true;
        }
    private:
        const Eigen::Vector2d pt;
        const double w;
    };

    struct WarpFunctorSimilarity{
    public:
        template<typename T>
        static Eigen::Matrix<T,2,1> getLocalCoord(const Eigen::Matrix<T,2,1>& p1, const Eigen::Matrix<T,2,1>& p2, const Eigen::Matrix<T,2,1>& p3){
            Eigen::Matrix<T,2,1> ax1 = p3 - p2;
            Eigen::Matrix<T,2,1> ax2(-1.0*ax1[1], ax1[0]);
            CHECK_GT(ax1.norm(), (T)0.0);
            Eigen::Matrix<T,2,1> uv;
            uv[0] = (p1-p2).dot(ax1) / ax1.norm() / ax1.norm();
            uv[1] = (p1-p2).dot(ax2) / ax2.norm() / ax2.norm();
            return uv;
        }

        WarpFunctorSimilarity(const Eigen::Vector2d& p1, const Eigen::Vector2d& p2, const Eigen::Vector2d& p3, const double w_): w(std::sqrt(w_)){
            refUV = getLocalCoord<double>(p1,p2,p3);
        }

        template<typename T>
        bool operator()(const T* const g1, const T* const g2, const T* const g3, T* residual)const{
            Eigen::Matrix<T,2,1> p1(g1[0], g1[1]);
            Eigen::Matrix<T,2,1> p2(g2[0], g2[1]);
            Eigen::Matrix<T,2,1> p3(g3[0], g3[1]);
            //Eigen::Matrix<T,2,1> curUV = getLocalCoord<T>(p1,p2,p3);
            Eigen::Matrix<T,2,1> axis1 = p3-p2;
            Eigen::Matrix<T,2,1> axis2(-1.0*axis1[1], axis1[0]);
            //Matrix<T,2,1> reconp1 = axis1 * refUV[0] + axis2 * refUV[1];
            T reconx = axis1[0] * refUV[0] + axis2[0] * refUV[1] + p2[0];
            T recony = axis1[1] * refUV[0] + axis2[1] * refUV[1] + p2[1];
            T diffx = reconx - g1[0];
            T diffy = recony - g1[1];
            residual[0] = ceres::sqrt(diffx * diffx + diffy * diffy + (T)1e-10) * w;
            return true;
        }
    private:
        Eigen::Vector2d refUV;
        const double w;
    };
}//namespace dynamic_stereo

#endif //DYNAMICSTEREO_GRIDENERGY_H
