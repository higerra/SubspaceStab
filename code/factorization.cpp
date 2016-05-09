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
								 Eigen::MatrixXd &bas, std::vector<bool>& is_computed, const int tWindow, const int stride) {
			const int kBasis = 9;
			const int kTrack = (int) trackMatrix.tracks.size();
			char buffer[1024] = {};
			const int N = (int)images.size();

			coe = MatrixXd(2 * kTrack, kBasis);
			bas = MatrixXd(kBasis, N);

			is_computed.resize(kTrack, false);
			vector<bool> is_valid(kTrack, true);

			const int testV = 0;
			//factorize the first window
			{
				vector<int> fullTrackInd;
				for (auto tid = 0; tid < kTrack; ++tid) {
					if (trackMatrix.offset[tid] <= testV) {
						if (trackMatrix.tracks[tid].size() >= tWindow)
							fullTrackInd.push_back(tid);
					}
				}
				printf("First window, complete track: %d\n", (int) fullTrackInd.size());

				//filterDynamicTrack(images, trackMatrix, fullTrackInd, testV, tWindow, is_valid);

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
					is_computed[idx] = true;
					coe.block(2 * idx, 0, 2, kBasis) = curcoe.block(2 * ftid, 0, 2, kBasis);
				}

				{
////				//sanity check
					//MatrixXd reconA = coe.block(0,0,(int)fullTrackInd.size()*2, coe.cols()) * bas.block(0,testV,kBasis,tWindow);
					MatrixXd reconA = curcoe * curbas;
					CHECK_EQ(reconA.cols(), A.cols());
					CHECK_EQ(reconA.rows(), A.rows());
					double reconError = 0.0;
					for (auto ftid = 0; ftid < fullTrackInd.size(); ++ftid) {
						const int idx = fullTrackInd[ftid];
						const int offset = (int) trackMatrix.offset[idx];
						for (auto i = testV; i < tWindow + testV; ++i) {
							Vector2d oriPt;
							oriPt[0] = trackMatrix.tracks[idx][i - offset].x;
							oriPt[1] = trackMatrix.tracks[idx][i - offset].y;
							Vector2d reconPt = reconA.block(ftid * 2, i - testV, 2, 1);
							reconError += (oriPt - reconPt).norm();
						}
					}
					printf("Reconstruction error for first window: %.3f\n",
						   reconError / ((double) fullTrackInd.size() * tWindow));
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
				printf("-------------------\nv: %d/%d\n", v, N);
				vector<int> preFullTrackInd;
				vector<int> newFullTrackInd;
				for (auto tid = 0; tid < kTrack; ++tid) {
					if(!is_valid[tid])
						continue;
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

//				filterDynamicTrack(images, trackMatrix, preFullTrackInd, v, tWindow, is_valid);
//				filterDynamicTrack(images, trackMatrix, newFullTrackInd, v, tWindow, is_valid);
//
//				for(auto tid=0; tid<is_valid.size(); ++tid){
//					if(!is_valid[tid])
//						is_computed[tid] = false;
//				}

				MatrixXd A12((int) preFullTrackInd.size() * 2, stride);
				MatrixXd A2((int) newFullTrackInd.size() * 2, tWindow);
				MatrixXd C1((int) preFullTrackInd.size() * 2, kBasis);
				MatrixXd A11((int) preFullTrackInd.size() * 2, tWindow - stride);

				MatrixXd E1 = bas.block(0, v, bas.rows(), tWindow - stride);

				for (auto ftid = 0; ftid < preFullTrackInd.size(); ++ftid) {
					const int idx = preFullTrackInd[ftid];
					CHECK(is_computed[idx]);
					const int offset = (int) trackMatrix.offset[idx];
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
					const int offset = (int) trackMatrix.offset[idx];
					for (auto i = v; i < v + tWindow; ++i) {
						A2(ftid * 2, i - v) = trackMatrix.tracks[idx][i - offset].x;
						A2(ftid * 2 + 1, i - v) = trackMatrix.tracks[idx][i - offset].y;
					}
				}

				MatrixXd A21 = A2.block(0, 0, A2.rows(), tWindow - stride);
				MatrixXd A22 = A2.block(0, tWindow - stride, A2.rows(), stride);

				MatrixXd C2(A21.rows(), kBasis);
				MatrixXd EE = E1 * E1.transpose();
				MatrixXd EEinv = EE.inverse();
				C2 = A21 * E1.transpose() * EEinv;

				MatrixXd largeC(C1.rows() + C2.rows(), C1.cols());
				largeC << C1,
						C2;
				MatrixXd largeA(A12.rows() + A22.rows(), A12.cols());
				largeA << A12,
						A22;
				MatrixXd E2 = (largeC.transpose() * largeC).inverse() * largeC.transpose() * largeA;
				bas.block(0, v + tWindow - stride, bas.rows(), stride) = E2;
				for (auto ftid = 0; ftid < newFullTrackInd.size(); ++ftid) {
					const int idx = newFullTrackInd[ftid];
					coe.block(2 * idx, 0, 2, coe.cols()) = C2.block(2 * ftid, 0, 2, C2.cols());
					is_computed[idx] = true;
				}


				{
					//sanity check: compute reconstruction error in current window
					MatrixXd reconA = C2 * bas.block(0, v, bas.rows(), tWindow);
					CHECK_EQ(reconA.rows(), A2.rows());
					CHECK_EQ(reconA.cols(), A2.cols());
					double newTrackError = 0.0;
					for (auto ftid = 0; ftid < newFullTrackInd.size(); ++ftid) {
						const int idx = newFullTrackInd[ftid];
						const int offset = (int) trackMatrix.offset[idx];
						for (auto i = v; i < v + tWindow; ++i) {
							Vector2d oriPt;
							oriPt[0] = trackMatrix.tracks[idx][i - offset].x;
							oriPt[1] = trackMatrix.tracks[idx][i - offset].y;
							Vector2d reconPt = reconA.block(2 * ftid, i - v, 2, 1);
							newTrackError += (reconPt - oriPt).norm();
						}
					}

					MatrixXd fullA = C1 * bas.block(0, v, bas.rows(), tWindow);
					double preFullError = 0.0;
					for (auto ftid = 0; ftid < preFullTrackInd.size(); ++ftid) {
						const int idx = preFullTrackInd[ftid];
						const int offset = (int) trackMatrix.offset[idx];
						for (auto i = v; i < v + tWindow; ++i) {
							Vector2d pt;
							pt[0] = trackMatrix.tracks[idx][i - offset].x;
							pt[1] = trackMatrix.tracks[idx][i - offset].y;
							Vector2d reconPt = fullA.block(2 * ftid, i - v, 2, 1);
							preFullError += (reconPt - pt).norm();
						}
					}
					printf("Reconstruction error: pre full:%.3f, new full:%.3f, overall: %.3f\n",
						   preFullError / ((double) preFullTrackInd.size() * tWindow),
						   newTrackError / ((double) newFullTrackInd.size() * tWindow),
						   (preFullError + newTrackError) /
						   (((double) preFullTrackInd.size() + (double) newFullTrackInd.size()) * tWindow));
				}

			}

		}


		void filterDynamicTrack(const std::vector<cv::Mat>& images, const FeatureTracks& trackMatrix, std::vector<int>& fullTrackInd,
								const int sf, const int tw, vector<bool>& is_valid){
			vector<cv::Point2f> pts1, pts2;
			for(auto ftid=0; ftid<fullTrackInd.size(); ++ftid){
				const int idx = fullTrackInd[ftid];
				const int offset = (int)trackMatrix.offset[idx];
				pts1.push_back(trackMatrix.tracks[idx][sf-offset]);
				pts2.push_back(trackMatrix.tracks[idx][sf+tw-1-offset]);
			}

			Mat fMatrix = cv::findFundamentalMat(pts1,pts2);
			Mat epilines;
			cv::computeCorrespondEpilines(pts1,1,fMatrix,epilines);

			vector<int> inlierInd;
			const double max_epiError = 2.0;
			for(auto ptid=0; ptid<pts1.size(); ++ptid){
				Vector3d pt(pts2[ptid].x, pts2[ptid].y, 1.0);
				Vector3d epi(epilines.at<Vec3f>(ptid,0)[0],epilines.at<Vec3f>(ptid,0)[1],epilines.at<Vec3f>(ptid,0)[2]);
				double epiError = pt.dot(epi);
				if(epiError >= max_epiError) {
					is_valid[fullTrackInd[ptid]] = false;
				}
				else {
					inlierInd.push_back(fullTrackInd[ptid]);
				}
			}

			{
//				char buffer[1024] = {};
//				Mat img1 = images[sf].clone();
//				Mat img2 = images[sf + tw - 1].clone();
//				const int testId = 100;
//				cv::circle(img1, pts1[testId], 2, Scalar(0,0,255), 2);
//				Vector3d testEpi(epilines.at<Vec3f>(testId,0)[0],epilines.at<Vec3f>(testId,0)[1],epilines.at<Vec3f>(testId,0)[2]);
//				for(auto y=0; y<img2.rows; ++y){
//					for(auto x=0; x<img2.cols; ++x){
//						Vector3d loc(x,y,1);
//						if(std::abs(loc.dot(testEpi)) < 1.0)
//							cv::circle(img2, cv::Point2i(x,y), 1, cv::Scalar(255,0,0), 1);
//					}
//				}
//				Mat himg;
//				cv::hconcat(img1, img2, himg);
//				sprintf(buffer, "epiTest_sf%d_tw%d_id%d.jpg", sf, tw, testId);
//				imwrite(buffer, himg);
			}

			printf("%d out of %d tracks are inliers\n", (int)inlierInd.size(), (int)fullTrackInd.size());
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
						if(k < 0 || k >= input.cols())
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
