//
// Created by yanhang on 5/4/16.
//
#include "factorization.h"
using namespace cv;
using namespace std;
using namespace Eigen;

namespace substab{

    void movingFactorization(const FeatureTracks& trackMatrix, cv::Mat& coe, cv::Mat& bas, const int N, const int tWindow, const int stride){
        const int kBasis = 9;
        const int kTrack = (int)trackMatrix.tracks.size();

	    bas = Mat(kBasis, N, CV_64F, Scalar::all(0));
	    coe = Mat(2*kTrack, kBasis, CV_64F, Scalar::all(0));

	    vector<bool> is_computed(kTrack, false);

	    for(auto v=0; v<N-tWindow; v+=stride) {
		    printf("==============\n");
		    printf("v:%d/%d\n", v, N);
		    vector<int> fullTrackInd;
		    vector<int> partialTrackInd;
		    //compute first window
		    for (auto tid = 0; tid < kTrack; ++tid) {
			    if (trackMatrix.offset[tid] <= v) {
				    if (trackMatrix.tracks[tid].size() + trackMatrix.offset[tid] >= v + tWindow)
					    fullTrackInd.push_back(tid);
				    else if (trackMatrix.offset[tid] + trackMatrix.tracks[tid].size() > v)
					    partialTrackInd.push_back(tid);
			    }
		    }

		    Mat A = Mat((int) fullTrackInd.size() * 2, tWindow, CV_64F, Scalar::all(0));

		    printf("Full track number: %d, partial track number: %d\n", (int)fullTrackInd.size(), (int)partialTrackInd.size());

		    for (auto ftid = 0; ftid < fullTrackInd.size(); ++ftid) {
			    const int idx = fullTrackInd[ftid];
			    for (auto i = 0; i < tWindow; ++i) {
				    A.at<double>(ftid * 2, i) = trackMatrix.tracks[idx][i+v-trackMatrix.offset[idx]].x;
				    A.at<double>(ftid * 2 + 1, i) = trackMatrix.tracks[fullTrackInd[ftid]][i+v-trackMatrix.offset[idx]].y;
			    }
		    }
		    Mat u, w, vt;
		    cv::SVD svd;
		    svd.compute(A, w, u, vt);
		    Mat w2 = cv::Mat::diag(w.rowRange(0, kBasis));
		    cv::sqrt(w2, w2);
		    Mat curcoe = u.colRange(0, kBasis) * w2;
		    Mat curbas = w2 * vt.rowRange(0, kBasis);
		    curbas.copyTo(bas.colRange(v,v+tWindow));

		    for(auto ftid=0; ftid < fullTrackInd.size(); ++ftid){
			    const int idx = fullTrackInd[ftid];
			    curcoe.row(ftid * 2).copyTo(coe.row(fullTrackInd[ftid] * 2));
			    curcoe.row(ftid * 2 + 1).copyTo(coe.row(fullTrackInd[ftid] * 2 + 1));
			    is_computed[fullTrackInd[ftid]] = true;
		    }

		    //compute coefficient for incomplete track
		    for(auto ftid=0; ftid < partialTrackInd.size(); ++ftid){
			    const int idx = partialTrackInd[ftid];
			    const int curL = (int)(trackMatrix.offset[idx]+trackMatrix.tracks[idx].size()-v);
			    Mat A21(2, curL, CV_64F, cv::Scalar::all(0));
			    for(auto i=0; i<curL; ++i){
				    A21.at<double>(0, i) = trackMatrix.tracks[idx][v+i-trackMatrix.offset[idx]].x;
				    A21.at<double>(1, i) = trackMatrix.tracks[idx][v+i-trackMatrix.offset[idx]].y;
			    }
			    Mat bas21 = curbas.colRange(0, curL);
			    Mat bas21T;
			    cv::transpose(bas21, bas21T);
			    Mat temp = bas21 * bas21T;
			    cv::invert(temp,temp);
			    Mat coe21 = A21 * bas21T * temp;
			    coe21.copyTo(coe.rowRange(2*idx, 2*idx+2));
			    is_computed[idx] = true;
		    }


		    Mat reconA = curcoe * curbas;
		    //Mat reconA = u * cv::Mat::diag(w) * vt;
		    printf("reconstruction error:%.3f\n", cv::norm(reconA - A) / (double) A.rows);
	    }

	    //compute overall approximation error
	    Mat recon = coe * bas;
	    CHECK_EQ(recon.rows, 2 * kTrack);
	    CHECK_EQ(recon.cols, N);
	    double accuError = 0.0;
	    double count = 0.0;
	    for(auto tid=0; tid<kTrack; ++tid){
		    for(auto i=trackMatrix.offset[tid]; i<trackMatrix.tracks[tid].size() + trackMatrix.offset[tid]; ++i){
			    cv::Point2f reconPt((float)recon.at<double>(2*tid, i), (float)recon.at<double>(2*tid+1, i));
			    accuError += cv::norm(reconPt - trackMatrix.tracks[tid][i-trackMatrix.offset[tid]]);
			    count += 1.0;
		    }
	    }
	    printf("Overall reconstruction error: %.3f\n", accuError / count);
    }

}//namespace substab
