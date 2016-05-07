//
// Created by yanhang on 5/4/16.
//
#include "factorization.h"
using namespace cv;
using namespace std;
using namespace Eigen;

namespace substab{

    void movingFactorization(const vector<Mat>& images, const FeatureTracks& trackMatrix, Eigen::MatrixXd& coe, Eigen::MatrixXd& bas, const int N, const int tWindow, const int stride) {
		const int kBasis = 9;
		const int kTrack = (int) trackMatrix.tracks.size();
	    char buffer[1024] = {};

		coe = MatrixXd(2*kTrack, kBasis);
		bas = MatrixXd(kBasis, N);

		vector<bool> is_computed(kTrack, false);

		//factorize the first window
		{
			vector<int> fullTrackInd;
			for (auto tid = 0; tid < kTrack; ++tid) {
				if (trackMatrix.offset[tid] == 0) {
					if (trackMatrix.tracks[tid].size() + trackMatrix.offset[tid] >= tWindow)
						fullTrackInd.push_back(tid);
				}
			}
			MatrixXd A((int)fullTrackInd.size()*2, tWindow);

			for (auto ftid = 0; ftid < fullTrackInd.size(); ++ftid) {
				const int idx = fullTrackInd[ftid];
				for (auto i = 0; i < tWindow; ++i) {
					A(ftid * 2, i) = trackMatrix.tracks[idx][i].x;
					A(ftid * 2 + 1, i) = trackMatrix.tracks[idx][i].y;
				}
			}

			Eigen::JacobiSVD<MatrixXd> svd(A, ComputeThinU|ComputeThinV);
			MatrixXd w = svd.singularValues().block(0,0,kBasis,1).asDiagonal();
			w = w.array().sqrt();
			MatrixXd curcoe = svd.matrixU().block(0,0,A.rows(),kBasis) * w;
			MatrixXd curbas = w * svd.matrixV().transpose().block(0,0,kBasis,A.cols());

			bas.block(0,0,kBasis,tWindow) = curbas;
			for(auto ftid=0; ftid<fullTrackInd.size(); ++ftid){
				const int idx = fullTrackInd[ftid];
				coe.block(2*idx,0,2,kBasis) = curcoe.block(2*ftid,0,2,kBasis);
			}

//			{
////				//sanity check
//				MatrixXd reconA = coe.block(0,0,(int)fullTrackInd.size()*2, coe.cols()) * bas.block(0,0,kBasis,tWindow);
//				CHECK_EQ(reconA.cols(), A.cols());
//				CHECK_EQ(reconA.rows(), A.rows());
//				double reconError = 0.0;
//				for (auto ftid = 0; ftid < fullTrackInd.size(); ++ftid) {
//					const int idx = fullTrackInd[ftid];
//					for (auto i = 0; i < tWindow; ++i) {
//						Vector2d oriPt;
//						oriPt[0] = trackMatrix.tracks[idx][i].x;
//						oriPt[1] = trackMatrix.tracks[idx][i].y;
//						Vector2d reconPt = reconA.block(ftid * 2, i, 2, 1);
//						reconError += (oriPt - reconPt).norm();
//					}
//				}
//				printf("Reconstruction error for first window: %.3f\n",
//					   reconError / ((double) fullTrackInd.size() * tWindow));
//			}
	    }

		//moving factorization
		for(auto v=stride; v<N-tWindow; v+=stride) {
			printf("v: %d/%d\n", v, N);
			vector<int> preFullTrackInd;
			vector<int> newFullTrackInd;
			int A11Count = 0;
			for (auto tid = 0; tid < kTrack; ++tid) {
				if (trackMatrix.offset[tid] <= v &&
					trackMatrix.offset[tid] + trackMatrix.tracks[tid].size() >= v + tWindow) {
					if (trackMatrix.offset[tid] <= v - stride) {
						//complete track inside previous window
						preFullTrackInd.push_back(tid);
					} else {
						//new complete track
						newFullTrackInd.push_back(tid);
					}
				}
			}

			printf("previous complete track:%d\n", (int)preFullTrackInd.size());
			printf("new complete track:%d\n", (int)newFullTrackInd.size());
			MatrixXd A12((int) preFullTrackInd.size() * 2, stride);
			MatrixXd A2((int) newFullTrackInd.size() * 2, tWindow);
			MatrixXd C1((int) preFullTrackInd.size() * 2, kBasis);
			MatrixXd A11((int) preFullTrackInd.size() * 2, tWindow - stride);

			MatrixXd E1 = bas.block(0, v, bas.rows(), tWindow - stride);

			for (auto ftid = 0; ftid < preFullTrackInd.size(); ++ftid) {
				const int idx = preFullTrackInd[ftid];
				const int offset = (int) trackMatrix.offset[idx];
				for (auto i = v; i < v + tWindow - stride; ++i) {
					A11(ftid * 2, i - v) = trackMatrix.tracks[idx][i - offset].x;
					A11(ftid * 2 + 1, i - v) = trackMatrix.tracks[idx][i - offset].y;
				}
				for (auto i = v + tWindow - stride; i < v + tWindow; ++i) {
					A12(ftid * 2, i - v - tWindow + stride) = trackMatrix.tracks[idx][i - offset].x;
					A12(ftid * 2 + 1, i - v - tWindow + stride) = trackMatrix.tracks[idx][i - offset].y;
				}
				for (auto i = 0; i < kBasis; ++i) {
					C1(ftid * 2, i) = coe(idx * 2, i);
					C1(ftid * 2 + 1, i) = coe(idx * 2 + 1, i);
				}
			}

			for (auto ftid = 0; ftid < newFullTrackInd.size(); ++ftid) {
				const int idx = newFullTrackInd[ftid];
				const int offset = (int)trackMatrix.offset[idx];
				for (auto i = v; i < v + tWindow; ++i) {
					A2(ftid * 2, i - v) = trackMatrix.tracks[idx][i-offset].x;
					A2(ftid * 2 + 1, i - v) = trackMatrix.tracks[idx][i-offset].y;
				}
			}

			MatrixXd A21 = A2.block(0, 0, A2.rows(), tWindow - stride);
			MatrixXd A22 = A2.block(0, tWindow - stride, A2.rows(), stride);

			{
				//sanity check: compute the reconstruction error from previous full track
				if (!preFullTrackInd.empty()) {
					MatrixXd fullA = C1 * E1;
					double preFullError = 0.0;
					for (auto ftid = 0; ftid < preFullTrackInd.size(); ++ftid) {
						const int idx = preFullTrackInd[ftid];
						for (auto i = v; i < v + tWindow - stride; ++i) {
//							Vector2d pt;
//							pt[0] = trackMatrix.tracks[idx][i-trackMatrix.offset[idx]].x;
//							pt[1] = trackMatrix.tracks[idx][i-trackMatrix.offset[idx]].y;
							Vector2d pt = A11.block(2*ftid,i-v,2,1);
							Vector2d reconPt = fullA.block(2 * ftid, i - v, 2, 1);
							preFullError += (reconPt - pt).norm();
						}
					}
					printf("Reconstruction error for previous complete track: %.3f\n",
						   preFullError / ((double) preFullTrackInd.size() * (tWindow - stride)));
				}
			}

			MatrixXd C2(A21.rows(), kBasis);
			MatrixXd EE = E1*E1.transpose();
			MatrixXd EEinv = EE.inverse();
			C2 = A21 * E1.transpose() * EEinv;

			MatrixXd largeC(C1.rows() + C2.rows(), C1.cols());
			largeC << C1,
					C2;
			MatrixXd largeA(A12.rows() + A22.rows(), A12.cols());
			largeA << A12,
					A22;
			MatrixXd E2 = (largeC.transpose() * largeC).inverse() * largeC.transpose() * largeA;
			bas.block(0, v+tWindow-stride, bas.rows(), stride) = E2;
			for(auto ftid=0; ftid<newFullTrackInd.size(); ++ftid){
				const int idx = newFullTrackInd[ftid];
				coe.block(2*idx,0,2,coe.cols()) = C2.block(2*ftid,0,2,C2.cols());
			}

//			cout << "E2:" << endl << E2 << endl;
//			cout << "C2:" << endl << C2 << endl;
			{
				//sanity check: compute reconstruction error in current window
				MatrixXd reconA = C2 * bas.block(0,v,bas.rows(), tWindow);
				CHECK_EQ(reconA.rows(), A2.rows());
				CHECK_EQ(reconA.cols(), A2.cols());
				double newTrackError = 0.0;
				for(auto ftid=0; ftid<newFullTrackInd.size(); ++ftid){
					const int idx = newFullTrackInd[ftid];
					const int offset = (int)trackMatrix.offset[idx];
					for(auto i=v; i<v+tWindow; ++i){
						Vector2d oriPt;
						oriPt[0] = trackMatrix.tracks[idx][i-offset].x;
						oriPt[1] = trackMatrix.tracks[idx][i-offset].y;
						Vector2d reconPt = reconA.block(2*ftid, i-v, 2, 1);
						newTrackError += (reconPt - oriPt).norm();
					}
				}
				printf("Reconstruction error for new tracks: %.3f\n", newTrackError / ((double)newFullTrackInd.size() * tWindow));
			}

		}





//	    for(auto v=0; v<N-tWindow; v+=stride) {
//		    printf("==============\n");
//		    printf("v:%d/%d\n", v, N);
//		    vector<int> fullTrackInd;
//		    vector<int> partialTrackInd;
//		    //compute first window
//		    for (auto tid = 0; tid < kTrack; ++tid) {
//			    if (trackMatrix.offset[tid] <= v) {
//				    if (trackMatrix.tracks[tid].size() + trackMatrix.offset[tid] >= v + tWindow)
//					    fullTrackInd.push_back(tid);
//				    else if (trackMatrix.offset[tid] + trackMatrix.tracks[tid].size() > v)
//					    partialTrackInd.push_back(tid);
//			    }
//		    }
//
//		    Mat A = Mat((int) fullTrackInd.size() * 2, tWindow, CV_64F, Scalar::all(0));
//
//		    printf("Full track number: %d, partial track number: %d\n", (int)fullTrackInd.size(), (int)partialTrackInd.size());
//
//		    for (auto ftid = 0; ftid < fullTrackInd.size(); ++ftid) {
//			    const int idx = fullTrackInd[ftid];
//			    for (auto i = 0; i < tWindow; ++i) {
//				    A.at<double>(ftid * 2, i) = trackMatrix.tracks[idx][i+v-trackMatrix.offset[idx]].x;
//				    A.at<double>(ftid * 2 + 1, i) = trackMatrix.tracks[fullTrackInd[ftid]][i+v-trackMatrix.offset[idx]].y;
//			    }
//		    }
//		    Mat u, w, vt;
//		    cv::SVD svd;
//		    svd.compute(A, w, u, vt);
//		    Mat w2 = cv::Mat::diag(w.rowRange(0, kBasis));
//		    cv::sqrt(w2, w2);
//		    Mat curcoe = u.colRange(0, kBasis) * w2;
//		    Mat curbas = w2 * vt.rowRange(0, kBasis);
////			if(v == 0)
////				curbas.copyTo(bas.colRange(v,v+tWindow));
////			else
////				curbas.colRange(tWindow-stride, tWindow).copyTo(bas.colRange(v+tWindow-stride,v+tWindow));
//			curbas.copyTo(bas.colRange(v,v+tWindow));
//
//		    for(auto ftid=0; ftid < fullTrackInd.size(); ++ftid){
//			    const int idx = fullTrackInd[ftid];
//			    if(is_computed[idx])
//				    continue;
//			    curcoe.row(ftid * 2).copyTo(coe.row(fullTrackInd[ftid] * 2));
//			    curcoe.row(ftid * 2 + 1).copyTo(coe.row(fullTrackInd[ftid] * 2 + 1));
//			    is_computed[fullTrackInd[ftid]] = true;
//		    }
//
//		    double fullReconError = 0.0;
//		    //Mat reconA = curcoe * curbas;
//			Mat reconA = curcoe * bas.colRange(v,v+tWindow);
//		    CHECK_EQ(A.size(), reconA.size());
//		    for(auto ftid=0; ftid<fullTrackInd.size(); ++ftid){
//			    for(auto i=0; i<tWindow; ++i){
//				    fullReconError += cv::norm(A(cv::Rect(i, 2*ftid, 1, 2)) - reconA(cv::Rect(i, 2*ftid,1,2)));
//			    }
//		    }
//		    printf("full track reconstruction error:%.3f\n", fullReconError / ((double)fullTrackInd.size() * tWindow));
//
//		    //compute coefficient for incomplete track
//		    double partialError = 0.0;
//		    double partialCount = 0.0;
//		    for(auto ftid=0; ftid < partialTrackInd.size(); ++ftid){
//			    const int idx = partialTrackInd[ftid];
//			    if(is_computed[idx])
//				    continue;
//			    const int curL = (int)(trackMatrix.offset[idx]+trackMatrix.tracks[idx].size()-v);
//			    Mat A21(2, curL, CV_64F, cv::Scalar::all(0));
//			    for(auto i=0; i<curL; ++i){
//				    A21.at<double>(0, i) = trackMatrix.tracks[idx][v+i-trackMatrix.offset[idx]].x;
//				    A21.at<double>(1, i) = trackMatrix.tracks[idx][v+i-trackMatrix.offset[idx]].y;
//			    }
//			    Mat bas21 = curbas.colRange(0, curL);
//			    Mat bas21T;
//			    cv::transpose(bas21, bas21T);
//			    Mat temp = bas21 * bas21T;
//			    cv::invert(temp,temp);
//			    Mat coe21 = A21 * bas21T * temp;
//			    coe21.copyTo(coe.rowRange(2*idx, 2*idx+2));
//			    Mat curpRecon = coe21 * curbas;
//			    for(auto i=0; i<curL; ++i){
//				    partialError += cv::norm(curpRecon(cv::Rect(i,0,1,2))-A21(cv::Rect(i,0,1,2)));
//				    partialCount++;
//			    }
//			    is_computed[idx] = true;
//		    }
//			if(partialCount > 0)
//				printf("partial track reconstruction error:%.3f\n", partialError / partialCount);
//
//
//		    //Mat reconA = u * cv::Mat::diag(w) * vt;
//		    //printf("reconstruction error:%.3f\n", cv::norm(reconA - A) / (double) A.rows);
//	    }
//
//	    //compute overall approximation error
//	    Mat recon = coe * bas;
//	    CHECK_EQ(recon.rows, 2 * kTrack);
//	    CHECK_EQ(recon.cols, N);
//	    double accuError = 0.0;
//	    double count = 0.0;
//	    for(auto tid=0; tid<kTrack; ++tid){
//		    CHECK(is_computed[tid]);
//		    for(auto i=trackMatrix.offset[tid]; i<trackMatrix.tracks[tid].size() + trackMatrix.offset[tid]; ++i){
//			    cv::Point2f reconPt((float)recon.at<double>(2*tid, (int)i), (float)recon.at<double>(2*tid+1, (int)i));
//			    const cv::Point2f oriPt = trackMatrix.tracks[tid][i-trackMatrix.offset[tid]];
//			    accuError += std::sqrt((reconPt.x-oriPt.x) * (reconPt.x-oriPt.x) + (reconPt.y-oriPt.y) * (reconPt.y-oriPt.y));
//			    count += 1.0;
//		    }
//	    }
//	    printf("Overall reconstruction error: %.3f\n", accuError / count);
    }

}//namespace substab
