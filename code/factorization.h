//
// Created by yanhang on 5/4/16.
//

#ifndef SUBSPACESTAB_FACTORIZATION_H
#define SUBSPACESTAB_FACTORIZATION_H

#include "tracking.h"

namespace substab{

    void movingFactorization(const FeatureTracks& trackMatrixs, Eigen::MatrixXd& coe, Eigen::MatrixXd& bas, const int N, const int tWindow, const int stride);

}//namespace substab
#endif //SUBSPACESTAB_FACTORIZATION_H
