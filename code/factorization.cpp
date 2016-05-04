//
// Created by yanhang on 5/4/16.
//
#include "factorization.h"
using namespace cv;
using namespace std;

namespace substab{

    void movingFactorization(FeatureTracks& trackMatrix, cv::Mat& coe, cv::Mat& bas, const int N, const int tWindow, const int stride){
        const int kBasis = 9;
        bas = Mat(kBasis, N, CV_32F, Scalar::all(0));
        coe = Mat((int)trackMatrix.tracks.size(), kBasis, CV_32F, Scalar::all(0));

        const int kTrack = (int)trackMatrix.tracks.size();
        //compute first window
        Mat A, A11, A12, A21, A22;
        vector<int> fullTrackInd;
        for(auto tid=0; tid<kTrack; ++tid){
            if(trackMatrix.offset[tid] == 0 && trackMatrix.tracks[tid].size() >= tWindow)
                fullTrackInd.push_back(tid);
        }

        A.release();
        A = Mat((int)fullTrackInd.size()*2, tWindow, CV_32F, Scalar::all(0));
        for(auto ftid=0; ftid<fullTrackInd.size(); ++ftid){
            for(auto v=0; v<tWindow; ++v){
                A.at<float>(ftid*2, v) = trackMatrix.tracks[fullTrackInd[ftid]][v].x;
                A.at<float>(ftid*2+1, v) = trackMatrix.tracks[fullTrackInd[ftid]][v].y;
            }
        }
        PCA pca(A, Mat(), PCA::DATA_AS_ROW, 9);
        Mat ev = pca.eigenvectors;
        ev.copyTo(bas.colRange(0,tWindow));
        pca.project(A).copyTo(coe.rowRange(0,A.rows));
        Mat repoA = coe.rowRange(0, A.rows) * bas.colRange(0, tWindow) + cv::repeat(pca.mean, A.rows, 1);

        Mat backp = pca.backProject(coe.rowRange(0,A.rows));
        printf("reprojection error:%.3f,%.3f\n", cv::norm(repoA-A)/(float)A.rows, cv::norm(backp-A)/(float)A.rows);
        //moving factorization

    }

}//namespace substab
