//
// Created by yanhang on 4/21/16.
//

#include "warping.h"
#include "gridenergy.h"
#include "utility.h"
#include <Eigen/Sparse>

using namespace std;
using namespace Eigen;
using namespace cv;

namespace substab{

	GridWarpping::GridWarpping(const int w, const int h, const int gw, const int gh) : width(w), height(h), gridW(gw), gridH(gh) {
		blockW = (double) width / gridW;
		blockH = (double) height / gridH;
		gridLoc.resize((size_t) (gridW + 1) * (gridH + 1));
		for (auto x = 0; x <= gridW; ++x) {
			for (auto y = 0; y <= gridH; ++y) {
				gridLoc[y * (gridW + 1) + x] = Eigen::Vector2d(blockW * x, blockH * y);
				if (x == gridW)
					gridLoc[y * (gridW + 1) + x][0] -= 1.1;
				if (y == gridH)
					gridLoc[y * (gridW + 1) + x][1] -= 1.1;
			}
		}
	}

	void GridWarpping::getGridIndAndWeight(const Eigen::Vector2d &pt, Eigen::Vector4i &ind,
										   Eigen::Vector4d &w) const {
		CHECK_LE(pt[0], width - 1);
		CHECK_LE(pt[1], height - 1);
		int x = (int) floor(pt[0] / blockW);
		int y = (int) floor(pt[1] / blockH);

		//////////////
		// 1--2
		// |  |
		// 4--3
		/////////////
		ind = Vector4i(y * (gridW + 1) + x, y * (gridW + 1) + x + 1, (y + 1) * (gridW + 1) + x + 1,
					   (y + 1) * (gridW + 1) + x);

		const double &xd = pt[0];
		const double &yd = pt[1];
		const double xl = gridLoc[ind[0]][0];
		const double xh = gridLoc[ind[2]][0];
		const double yl = gridLoc[ind[0]][1];
		const double yh = gridLoc[ind[2]][1];

		w[0] = (xh - xd) * (yh - yd);
		w[1] = (xd - xl) * (yh - yd);
		w[2] = (xd - xl) * (yd - yl);
		w[3] = (xh - xd) * (yd - yl);

		double s = w[0] + w[1] + w[2] + w[3];
		CHECK_GT(s, 0) << pt[0] << ' '<< pt[1];
		w = w / s;

		Vector2d pt2 =
				gridLoc[ind[0]] * w[0] + gridLoc[ind[1]] * w[1] + gridLoc[ind[2]] * w[2] + gridLoc[ind[3]] * w[3];
		double error = (pt2 - pt).norm();
		CHECK_LT(error, 0.0001) << pt[0] << ' ' << pt[1] << ' ' << pt2[0] << ' ' << pt2[1];
	}


	void GridWarpping::visualizeGrid(const std::vector<Eigen::Vector2d>& grid, cv::Mat &img) const {
		CHECK_EQ(grid.size(), gridLoc.size());
		img = Mat(height, width, CV_8UC3, Scalar(0,0,0));
		for(auto gy=0; gy<gridH; ++gy) {
			for (auto gx = 0; gx < gridW; ++gx){
				const int gid1 = gy * (gridW+1) + gx;
				const int gid2 = (gy+1) * (gridW+1) + gx;
				const int gid3 = (gy+1)*(gridW+1)+gx+1;
				const int gid4= gy * (gridW+1) + gx+1;
				cv::line(img, cv::Point(grid[gid1][0], grid[gid1][1]), cv::Point(grid[gid2][0], grid[gid2][1]), Scalar(255,255,255));
				cv::line(img, cv::Point(grid[gid2][0], grid[gid2][1]), cv::Point(grid[gid3][0], grid[gid3][1]), Scalar(255,255,255));
				cv::line(img, cv::Point(grid[gid3][0], grid[gid3][1]), cv::Point(grid[gid4][0], grid[gid4][1]), Scalar(255,255,255));
				cv::line(img, cv::Point(grid[gid4][0], grid[gid4][1]), cv::Point(grid[gid1][0], grid[gid1][1]), Scalar(255,255,255));
			}
		}
	}

	void GridWarpping::warpImage(const cv::Mat& input, cv::Mat& output, const std::vector<Eigen::Vector2d>& pts1, const std::vector<Eigen::Vector2d>& pts2)const {
		CHECK_EQ(pts1.size(), pts2.size());
		CHECK_EQ(input.cols, width);
		CHECK_EQ(input.rows, height);

		char buffer[1024] = {};

		vector<vector<double> > vars(gridLoc.size());
		for(auto &v: vars)
			v.resize(2);

		vector<Vector2d> grid2(gridLoc.size());
		for (auto i = 0; i < gridLoc.size(); ++i) {
			vars[i][0] = gridLoc[i][0];
			vars[i][1] = gridLoc[i][1];
			grid2[i] = gridLoc[i];
		}

		const double wdata = 1.0;
		const double wsimilarity = 1.0;

		ceres::Problem problem;
		//data term
		for (auto i = 0; i < pts2.size(); ++i) {
			if(pts2[i][0] < 0 || pts2[i][1] < 0 || pts2[i][0] >= width-1 || pts2[i][1] >= height-1)
				continue;
			Vector4i indRef;
			Vector4d bwRef;
			getGridIndAndWeight(pts2[i], indRef, bwRef);
			problem.AddResidualBlock(
					new ceres::AutoDiffCostFunction<WarpFunctorData, 1, 2, 2, 2, 2>(
							new WarpFunctorData(pts1[i], bwRef, wdata)),
					NULL,
					vars[indRef[0]].data(), vars[indRef[1]].data(), vars[indRef[2]].data(), vars[indRef[3]].data()
			);
		}

		//simiarity term
		for (auto y = 1; y <= gridH; ++y) {
			for (auto x = 0; x < gridW; ++x) {
				int gid1, gid2, gid3;
				gid1 = y * (gridW + 1) + x;
				gid2 = (y - 1) * (gridW + 1) + x;
				gid3 = y * (gridW + 1) + x + 1;
				problem.AddResidualBlock(
						new ceres::AutoDiffCostFunction<WarpFunctorSimilarity, 1, 2, 2, 2>(
								new WarpFunctorSimilarity(gridLoc[gid1], gridLoc[gid2], gridLoc[gid3], wsimilarity)),
						NULL,
						vars[gid1].data(), vars[gid2].data(), vars[gid3].data()
				);
				gid2 = (y - 1) * (gridW + 1) + x + 1;
				problem.AddResidualBlock(
						new ceres::AutoDiffCostFunction<WarpFunctorSimilarity, 1, 2, 2, 2>(
								new WarpFunctorSimilarity(gridLoc[gid1], gridLoc[gid2], gridLoc[gid3], wsimilarity)),
						NULL,
						vars[gid1].data(), vars[gid2].data(), vars[gid3].data());
			}
		}

		ceres::Solver::Options options;
		options.max_num_iterations = 1000;
		options.linear_solver_type = ceres::SPARSE_NORMAL_CHOLESKY;
		ceres::Solver::Summary summary;
		ceres::Solve(options, &problem, &summary);
		cout << summary.BriefReport() << endl;


		//warping
		output = Mat(height, width, CV_8UC3, Scalar::all(0));
		for(auto y=0; y<height; ++y){
			for(auto x=0; x<width; ++x){
				Vector4i ind;
				Vector4d w;
				getGridIndAndWeight(Vector2d(x,y), ind, w);
				Vector2d pt(0,0);
				for(auto i=0; i<4; ++i){
					pt[0] += vars[ind[i]][0] * w[i];
					pt[1] += vars[ind[i]][1] * w[i];
				}
				if(pt[0] < 0 || pt[1] < 0 || pt[0] > width - 1 || pt[1] > height - 1)
					continue;
				Vector3d pixO = interpolation_util::bilinear<uchar,3>(input.data, input.cols, input.rows, pt);
				output.at<Vec3b>(y,x) = Vec3b((uchar) pixO[0], (uchar)pixO[1], (uchar)pixO[2]);
			}
		}
	}

	void GridWarpping::computeSimilarityWeight(const cv::Mat &input, std::vector<Eigen::Vector2d>& saliency) const {
		saliency.resize((size_t)((gridW+1)*(gridH+1)));
		vector<vector<Vector3d> > pixs(saliency.size());
		for(auto y=0; y<input.rows; ++y){
			for(auto x=0; x<input.cols; ++x){
				Vector4i ind;

			}
		}
	}


//	void GridWarpping::warpImageCloseForm(const cv::Mat &input, cv::Mat &output, const vector<Vector2d>& pts1, const vector<Vector2d>& pts2) const {
//		CHECK_EQ(pts1.size(), pts2.size());
//
//		vector<Vector2d> resGrid(gridLoc.size());
//		const int kDataTerm = (int)pts1.size() * 2;
//		const int kSimTerm = (gridW-1)*(gridH-1)*8;
//		const int kVar = (int)gridLoc.size() * 2;
//
//		vector<Eigen::Triplet<double> > triplets;
//		VectorXd B(kDataTerm+kSimTerm);
//		//add data constraint
//		printf("Creating matrix...\n");
//		const double wdata = 1.0;
//		const double wsimilarity = 1.0;
//		int cInd = 0;
//		printf("data term\n");
//		for(auto i=0; i<pts2.size(); ++i) {
//			if (pts2[i][0] < 0 || pts2[i][1] < 0 || pts2[i][0] >= width - 1 || pts2[i][1] >= height - 1)
//				continue;
//			Vector4i indRef;
//			Vector4d bwRef;
//			CHECK_LT(cInd + 1, B.rows());
//
//			getGridIndAndWeight(pts2[i], indRef, bwRef);
//			for (auto j = 0; j < 4; ++j) {
//				CHECK_LT(indRef[j]*2+1, kVar);
//				triplets.push_back(Triplet<double>(cInd, indRef[j] * 2, wdata * bwRef[j]));
//				triplets.push_back(Triplet<double>(cInd + 1, indRef[j] * 2 + 1, wdata * bwRef[j]));
//			}
//			B[cInd] = wdata * pts1[i][0];
//			B[cInd + 1] = wdata * pts1[i][1];
//			cInd += 2;
//		}
//
//		//add similarity constraint
//		auto gridInd = [&](int x, int y){
//			CHECK_LE(x,gridW);
//			CHECK_LE(y,gridH);
//			return y*(gridW+1)+x;
//		};
//		printf("similarity term\n");
//		for(auto y=1; y< gridH; ++y) {
//			for (auto x = 1; x < gridW; ++x) {
//				CHECK_LT(cInd+7, B.rows());
//				vector<Vector2i> gids{
//						Vector2i(gridInd(x-1,y), gridInd(x,y-1)),
//						Vector2i(gridInd(x,y-1), gridInd(x+1,y)),
//						Vector2i(gridInd(x+1,y), gridInd(x,y+1)),
//						Vector2i(gridInd(x,y+1), gridInd(x-1,y))
//				};
//				const int cgid = gridInd(x,y);
//				CHECK_LT(cgid, gridLoc.size());
//				for(const auto& gid: gids){
//					CHECK_LT(gid[0], gridLoc.size());
//					CHECK_LT(gid[1], gridLoc.size());
//					CHECK_LT(gid[0]*2+1, kVar);
//					CHECK_LT(gid[1]*2+1, kVar);
//
//					Vector2d refUV = WarpFunctorSimilarity::getLocalCoord<double>(gridLoc[cgid], gridLoc[gid[0]], gridLoc[gid[1]]);
//					//x coordinate
//					triplets.push_back(Triplet<double>(cInd, cgid*2, wsimilarity));
//					triplets.push_back(Triplet<double>(cInd, gid[0]*2, (refUV[0]-1) * wsimilarity));
//					triplets.push_back(Triplet<double>(cInd, gid[1]*2, -1 * refUV[0] * wsimilarity));
//					triplets.push_back(Triplet<double>(cInd, gid[0]*2+1, refUV[1] * wsimilarity));
//					triplets.push_back(Triplet<double>(cInd, gid[1]*2+1, -1 * refUV[1] * wsimilarity));
//					B[cInd] = 0;
//					//y coordinate
//					triplets.push_back(Triplet<double>(cInd+1, cgid*2+1, wsimilarity));
//					triplets.push_back(Triplet<double>(cInd+1, gid[0]*2+1, (refUV[0]-1)* wsimilarity));
//					triplets.push_back(Triplet<double>(cInd+1, gid[1]*2+1, -1 * refUV[0] * wsimilarity));
//					triplets.push_back(Triplet<double>(cInd+1, gid[0]*2, -1 * refUV[1] * wsimilarity));
//					triplets.push_back(Triplet<double>(cInd+1, gid[1]*2, refUV[1] * wsimilarity));
//					B[cInd+1] = 0;
//					cInd += 2;
//				}
//			}
//		}
//		CHECK_EQ(cInd, kDataTerm+kSimTerm);
//		printf("set from triplets, %d...\n", (int)triplets.size());
//		SparseMatrix<double> A(cInd, kVar);
//		A.setFromTriplets(triplets.begin(), triplets.end());
//		printf("done\n");
//		//solve
//		printf("%d,%d\n", A.rows(), A.cols());
//
//		//SimplicialLDLT<SparseMatrix<double> > solver(A);
//		//Eigen::SimplicialCholesky<SparseMatrix<double> > lesolver(A);
//		SparseQR<SparseMatrix<double>, COLAMDOrdering<int> > solver(A);
//		printf("Solving...\n");
//		VectorXd x = solver.solve(B.block(0,0,cInd,1));
//		printf("Done\n");
//		//CHECK_EQ(x.rows(), kVar);
//
//	}

}//namespace substablas
