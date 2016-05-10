//
// Created by yanhang on 5/4/16.
//
#include "factorization.h"
using namespace cv;
using namespace std;
using namespace Eigen;

namespace substab{
	namespace Factorization {
		void movingFactorization(const vector<Mat> &images, const FeatureTracks &trackMatrix, Eigen::MatrixXd &coe,
								 Eigen::MatrixXd &bas, vector<vector<int> >& wMatrix, const int tWindow, const int stride) {
			const int kBasis = 9;
			const int kTrack = (int) trackMatrix.tracks.size();
			char buffer[1024] = {};
			const int N = (int)images.size();

			vector<bool> is_computed(kTrack, false);

			coe = MatrixXd::Zero(2 * kTrack, kBasis);
			bas = MatrixXd::Zero(kBasis, N);

			const int testV = 0;
			//factorize the first window
			{
				vector<int> fullTrackInd;
				for (auto tid = 0; tid < kTrack; ++tid) {
					if (trackMatrix.offset[tid] <= testV) {
						if (trackMatrix.offset[tid] + trackMatrix.tracks[tid].size() >= tWindow + testV)
							fullTrackInd.push_back(tid);
					}
				}
//				printf("First window, complete track: %d\n", (int) fullTrackInd.size());

				filterDynamicTrack(images, trackMatrix, fullTrackInd, testV, tWindow, wMatrix);

				MatrixXd A((int) fullTrackInd.size() * 2, tWindow);

				for (auto ftid = 0; ftid < fullTrackInd.size(); ++ftid) {
					const int idx = fullTrackInd[ftid];
					const int offset = (int) trackMatrix.offset[idx];
					for (auto i = testV; i < tWindow + testV; ++i) {
						A(ftid * 2, i - testV) = trackMatrix.tracks[idx][i - offset].x;
						A(ftid * 2 + 1, i - testV) = trackMatrix.tracks[idx][i - offset].y;
					}
				}

				Eigen::JacobiSVD<MatrixXd> svd(A, ComputeThinU | ComputeThinV);
				MatrixXd w = svd.singularValues().block(0, 0, kBasis, 1).asDiagonal();
				w = w.array().sqrt();
				MatrixXd curcoe = svd.matrixU().block(0, 0, A.rows(), kBasis) * w;
				MatrixXd curbas = w * svd.matrixV().transpose().block(0, 0, kBasis, A.cols());

				bas.block(0, testV, kBasis, tWindow) = curbas;
				for (auto ftid = 0; ftid < fullTrackInd.size(); ++ftid) {
					const int idx = fullTrackInd[ftid];
					coe.block(2 * idx, 0, 2, kBasis) = curcoe.block(2 * ftid, 0, 2, kBasis);
					for(auto i=testV; i<testV+tWindow; ++i)
						wMatrix[idx][i] = 1;
					is_computed[idx] = true;
				}

				{
////				//sanity check
					//MatrixXd reconA = coe.block(0,0,(int)fullTrackInd.size()*2, coe.cols()) * bas.block(0,testV,kBasis,tWindow);
//					MatrixXd reconA = curcoe * curbas;
//					CHECK_EQ(reconA.cols(), A.cols());
//					CHECK_EQ(reconA.rows(), A.rows());
//					double reconError = 0.0;
//					double count = 0.0;
//					for (auto ftid = 0; ftid < fullTrackInd.size(); ++ftid) {
//						const int idx = fullTrackInd[ftid];
//						const int offset = (int) trackMatrix.offset[idx];
//						for (auto i = testV; i < tWindow + testV; ++i) {
//							Vector2d oriPt;
//							CHECK_LT(i-offset, trackMatrix.tracks[idx].size()) << i << ' ' << offset;
//							CHECK_GE(i-offset, 0) << i << ' ' << offset;
//							oriPt[0] = trackMatrix.tracks[idx][i - offset].x;
//							oriPt[1] = trackMatrix.tracks[idx][i - offset].y;
//							Vector2d reconPt = reconA.block(ftid * 2, i - testV, 2, 1);
//							count += 1.0;
//							reconError += (oriPt - reconPt).norm();
//						}
//					}
//					printf("Reconstruction error for first window: %.3f/%d=%.3f\n",
//						   reconError,(int)count, reconError / count);
//
//					for (auto i = testV; i < testV + tWindow; ++i) {
//						Mat outImg = images[i].clone();
//						for (auto ftid = 0; ftid < fullTrackInd.size(); ++ftid) {
//							const int idx = fullTrackInd[ftid];
//							const int offset = (int) trackMatrix.offset[idx];
//							cv::circle(outImg, trackMatrix.tracks[idx][i - offset], 1, Scalar(255, 0, 0), 2);
//							cv::circle(outImg,
//									   cv::Point2d(reconA(2 * ftid, i - testV), reconA(2 * ftid + 1, i - testV)), 1,
//									   Scalar(0, 0, 255), 2);
//						}
//						imshow("testV", outImg);
//						waitKey();
//					}

				}
			}

			//moving factorization
			for (auto v = stride + testV; v < N - tWindow; v += stride) {
//				printf("-------------------\nv: %d/%d\n", v, N);
				vector<int> preFullTrackInd;
				vector<int> newFullTrackInd;
				for (auto tid = 0; tid < kTrack; ++tid) {
					if (trackMatrix.offset[tid] <= v &&
						trackMatrix.offset[tid] + trackMatrix.tracks[tid].size() >= v + tWindow) {
						bool is_valid = true;
						for(auto i=v; i<v+tWindow; ++i){
							if(wMatrix[tid][i] == -1) {
								is_valid = false;
								break;
							}
						}
						if(!is_valid)
							continue;
						if (trackMatrix.offset[tid] <= v - stride) {
							//complete track inside previous window
							CHECK(is_computed[tid]);
							preFullTrackInd.push_back(tid);
						} else {
							//new complete track
							newFullTrackInd.push_back(tid);
						}
					}
				}

				filterDynamicTrack(images, trackMatrix, preFullTrackInd, v, tWindow, wMatrix);
				filterDynamicTrack(images, trackMatrix, newFullTrackInd, v, tWindow, wMatrix);

				MatrixXd A12 = MatrixXd::Zero((int) preFullTrackInd.size() * 2, stride);
				MatrixXd A2 = MatrixXd::Zero((int) newFullTrackInd.size() * 2, tWindow);
				MatrixXd C1 = MatrixXd::Zero((int) preFullTrackInd.size() * 2, kBasis);
				MatrixXd A11 = MatrixXd::Zero((int) preFullTrackInd.size() * 2, tWindow - stride);

				MatrixXd E1 = bas.block(0, v, bas.rows(), tWindow - stride);

				for (auto ftid = 0; ftid < preFullTrackInd.size(); ++ftid) {
					const int idx = preFullTrackInd[ftid];
					const int offset = (int) trackMatrix.offset[idx];
					CHECK(is_computed[idx]);
					for (auto i = v + tWindow - stride; i < v + tWindow; ++i) {
						CHECK_GE(i-offset, 0);
						CHECK_LT(i-offset, trackMatrix.tracks[idx].size());
						A12(ftid * 2, i - v - tWindow + stride) = trackMatrix.tracks[idx][i - offset].x;
						A12(ftid * 2 + 1, i - v - tWindow + stride) = trackMatrix.tracks[idx][i - offset].y;
					}
					C1.block(ftid*2, 0, 2, C1.cols()) = coe.block(idx*2,0,2,coe.cols());
				}

//				{
//					for(auto i=v+tWindow-stride; i<v+tWindow; ++i){
//						Mat img = images[i].clone();
//						for(auto ftid=0; ftid<preFullTrackInd.size(); ++ftid){
//							cv::circle(img, cv::Point2d(A12(ftid*2, i-v-tWindow+stride),A12(ftid*2+1, i-v-tWindow+stride)), 1, Scalar(0,0,255), 1);
//						}
//						imshow("A12", img);
//						waitKey();
//					}
//				}

				for (auto ftid = 0; ftid < newFullTrackInd.size(); ++ftid) {
					const int idx = newFullTrackInd[ftid];
					const int offset = (int) trackMatrix.offset[idx];
					for (auto i = v; i < v + tWindow; ++i) {
						CHECK_GE(i-offset, 0);
						CHECK_LT(i-offset, trackMatrix.tracks[idx].size());
						A2(ftid * 2, i - v) = trackMatrix.tracks[idx][i - offset].x;
						A2(ftid * 2 + 1, i - v) = trackMatrix.tracks[idx][i - offset].y;
					}
				}

				MatrixXd A21 = A2.block(0, 0, A2.rows(), tWindow - stride);
				MatrixXd A22 = A2.block(0, tWindow - stride, A2.rows(), stride);

				MatrixXd EE = E1 * E1.transpose();
				MatrixXd EEinv = EE.inverse();
				MatrixXd C2 = A21 * E1.transpose() * EEinv;

				MatrixXd largeC(C1.rows() + C2.rows(), C1.cols());
				largeC.block(0,0,C1.rows(),kBasis) = C1;
				largeC.block(C1.rows(),0,C2.rows(), kBasis) = C2;

				MatrixXd largeA(A12.rows() + A22.rows(), A12.cols());
				largeA.block(0,0,A12.rows(),A12.cols()) = A12;
				largeA.block(A12.rows(),0,A22.rows(), A22.cols()) = A22;

				MatrixXd E2 = (largeC.transpose() * largeC).inverse() * largeC.transpose() * largeA;
				//MatrixXd E2 = (C1.transpose() * C1).inverse() * C1.transpose() * A12;
				//MatrixXd E2 = (C2.transpose() * C2).inverse() * C2.transpose() * A22;

				bas.block(0, v + tWindow - stride, bas.rows(), stride) = E2;

				for (auto ftid = 0; ftid < newFullTrackInd.size(); ++ftid) {
					const int idx = newFullTrackInd[ftid];
					coe.block(2 * idx, 0, 2, coe.cols()) = C2.block(2 * ftid, 0, 2, C2.cols());
					for(auto i=v; i< v+ tWindow; ++i)
						wMatrix[idx][i] = 1;
					is_computed[idx] = true;
				}

				{
					//error A12
//					double errorA12 = 0.0, countA12 = 0.0;
//					MatrixXd reconA12 = C1 * bas.block(0,v+tWindow-stride,kBasis,stride);
//					for(auto ftid=0; ftid<preFullTrackInd.size(); ++ftid){
//						for(auto i=0; i<stride; ++i){
//							Vector2d oriPt = A12.block(ftid*2,i,2,1);
//							Vector2d reconPt = reconA12.block(ftid*2, i, 2,1);
//							errorA12 += (oriPt-reconPt).norm();
//							countA12 += 1.0;
//						}
//					}
//					printf("Reconstruction error for A12: %.3f\n", errorA12 / countA12);
//
//					//error A21
//					double errorA21 = 0.0, countA21 = 0.0;
//					MatrixXd reconA21 = C2 * bas.block(0,v,kBasis,tWindow-stride);
//					for(auto ftid=0; ftid<newFullTrackInd.size(); ++ftid){
//						for(auto i=0; i<tWindow-stride; ++i){
//							Vector2d oriPt = A21.block(ftid*2,i,2,1);
//							Vector2d reconPt = reconA21.block(ftid*2, i, 2,1);
//							errorA21 += (oriPt-reconPt).norm();
//							countA21 += 1.0;
//						}
//					}
//					printf("Reconstruction error for A21: %.3f\n", errorA21 / countA21);
//
//					//error A22
//					double errorA22 = 0.0, countA22 = 0.0;
//					MatrixXd reconA22 = C2 * bas.block(0,v+tWindow-stride,kBasis,stride);
//					for(auto ftid=0; ftid<newFullTrackInd.size(); ++ftid){
//						for(auto i=0; i<stride; ++i){
//							Vector2d oriPt = A22.block(ftid*2,i,2,1);
//							Vector2d reconPt = reconA22.block(ftid*2, i, 2,1);
//							errorA22 += (oriPt-reconPt).norm();
//							countA22 += 1.0;
//						}
//					}
//					printf("Reconstruction error for A22: %.3f\n", errorA22 / countA22);
//
//					//sanity check: compute reconstruction error in current window
//					MatrixXd reconA = C2 * bas.block(0, v, bas.rows(), tWindow);
//					CHECK_EQ(reconA.rows(), A2.rows());
//					CHECK_EQ(reconA.cols(), A2.cols());
//					double newTrackError = 0.0;
//					double countNewTrack = 0.0;
//					for (auto ftid = 0; ftid < newFullTrackInd.size(); ++ftid) {
//						const int idx = newFullTrackInd[ftid];
//						const int offset = (int) trackMatrix.offset[idx];
//						for (auto i = v; i < v + tWindow; ++i) {
//							Vector2d oriPt;
//							CHECK_GE(i-offset, 0);
//							CHECK_LT(i-offset, trackMatrix.tracks[idx].size());
//							oriPt[0] = trackMatrix.tracks[idx][i - offset].x;
//							oriPt[1] = trackMatrix.tracks[idx][i - offset].y;
//							Vector2d reconPt = reconA.block(2 * ftid, i - v, 2, 1);
//							newTrackError += (reconPt - oriPt).norm();
//							countNewTrack += 1.0;
//						}
//					}
//
//					MatrixXd fullA = C1 * bas.block(0, v-stride, bas.rows(), tWindow);
//					double preFullError = 0.0;
//					double countPreTrack = 0.0;
//					for (auto ftid = 0; ftid < preFullTrackInd.size(); ++ftid) {
//						const int idx = preFullTrackInd[ftid];
//						const int offset = (int) trackMatrix.offset[idx];
//						for (auto i = v-stride; i < v + tWindow-stride; ++i) {
//							Vector2d pt;
//							CHECK_GE(i-offset, 0);
//							CHECK_LT(i-offset, trackMatrix.tracks[idx].size());
//							pt[0] = trackMatrix.tracks[idx][i - offset].x;
//							pt[1] = trackMatrix.tracks[idx][i - offset].y;
//							Vector2d reconPt = fullA.block(2 * ftid, i - v + stride, 2, 1);
//							preFullError += (reconPt - pt).norm();
//							countPreTrack += 1.0;
//						}
//					}
//
//					double preFullError2 = 0.0;
//					double countPreTrack2 = 0.0;
//					MatrixXd fullA2 = C1 * bas.block(0, v, bas.rows(), tWindow);
//					for(auto ftid=0; ftid < preFullTrackInd.size(); ++ftid){
//						const int idx = preFullTrackInd[ftid];
//						const int offset = (int)trackMatrix.offset[idx];
//						for (auto i = v; i < v + tWindow; ++i) {
//							Vector2d pt;
//							CHECK_GE(i-offset, 0);
//							CHECK_LT(i-offset, trackMatrix.tracks[idx].size());
//							pt[0] = trackMatrix.tracks[idx][i - offset].x;
//							pt[1] = trackMatrix.tracks[idx][i - offset].y;
//							Vector2d reconPt = fullA2.block(2 * ftid, i - v, 2, 1);
//							preFullError2 += (reconPt - pt).norm();
//							countPreTrack2 += 1.0;
//						}
//					}
//
//					printf("Reconstruction error: pre full:%.3f, pre full2: %.3f, new full:%.3f, overall: %.3f\n",
//						   preFullError / countPreTrack,
//						   preFullError2 / countPreTrack2,
//						   newTrackError / countNewTrack,
//						   (preFullError2 + newTrackError) / (countPreTrack2 + countNewTrack));
				}
			}

			//compute overall error

		}


		void filterDynamicTrack(const std::vector<cv::Mat>& images, const FeatureTracks& trackMatrix, std::vector<int>& fullTrackInd,
								const int sf, const int tw, vector<vector<int> >& wMatrix){
			const double max_ratio = 0.33;
			vector<double> totalCount(fullTrackInd.size(), 0.0);
			vector<double> outlierCount(fullTrackInd.size(), 0.0);
			const double max_epiError = 2.0;
			const int stride = 5;

			for(auto v=0; v<sf+tw-stride; v+=stride) {
				vector<cv::Point2f> pts1, pts2;
				vector<size_t> trackId;
				for (auto ftid = 0; ftid < fullTrackInd.size(); ++ftid) {
					const int idx = fullTrackInd[ftid];
					const int offset = (int) trackMatrix.offset[idx];
					if(offset <= v && offset+trackMatrix.tracks[idx].size() >= v+stride) {
						pts1.push_back(trackMatrix.tracks[idx][v]);
						pts2.push_back(trackMatrix.tracks[idx][v+stride-1]);
						trackId.push_back(ftid);
					}
				}
				if(trackId.size() < 8)
					continue;
				Mat fMatrix = cv::findFundamentalMat(pts1, pts2);
				if(fMatrix.cols != 3)
					continue;
				Mat epilines;
				cv::computeCorrespondEpilines(pts1, 1, fMatrix, epilines);
				for(auto ptid=0; ptid<trackId.size(); ++ptid){
					Vector3d epi(epilines.at<Vec3f>(ptid,0)[0], epilines.at<Vec3f>(ptid,0)[1], epilines.at<Vec3f>(ptid,0)[2]);
					Vector3d pt(pts2[ptid].x, pts2[ptid].y, 1.0);
					totalCount[trackId[ptid]] += 1.0;
					if(epi.dot(pt) > max_epiError)
						outlierCount[trackId[ptid]] += 1.0;
				}
			}

			vector<int> inlierInd;

			for(auto i=0; i<fullTrackInd.size(); ++i){
				if(outlierCount[i] / max_ratio <= totalCount[i])
					inlierInd.push_back(fullTrackInd[i]);
				else{
					for(auto v=sf; v<wMatrix[fullTrackInd[i]].size(); ++v)
						wMatrix[fullTrackInd[i]][v] = -1;
				}
			}
			fullTrackInd.swap(inlierInd);
		}

		void trackSmoothing(const Eigen::MatrixXd& input, Eigen::MatrixXd& output, const int r, const double sigma){
			Mat gauKernelCV = cv::getGaussianKernel(2*r+1,sigma,CV_64F);
			const double* pKernel = (double*) gauKernelCV.data;
			output = MatrixXd::Zero(input.rows(), input.cols());
			for(auto i=0; i<input.rows(); ++i){
				for(auto j=0; j<input.cols(); ++j){
					double sum = 0.0;
					for(auto k=-1*r; k<=r; ++k){
						if(j+k < 0 || j+k >= input.cols())
							continue;
						sum += pKernel[k+r];
						output(i,j) += input(i,j+k) * pKernel[k+r];
					}
					output(i,j) /= sum;
				}
			}
		}

	}//namespace Factorization
}//namespace substab
