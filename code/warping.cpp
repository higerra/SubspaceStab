//
// Created by yanhang on 4/21/16.
//

#include "warping.h"
#include "gridenergy.h"
#include "utility.h"
#include <Eigen/Sparse>
#include <Eigen/SPQRSupport>
#include <fstream>

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
		CHECK_EQ(img.cols, width);
		CHECK_EQ(img.rows, height);
		//img = Mat(height, width, CV_8UC3, Scalar(0,0,0));
		for(auto gy=0; gy<gridH; ++gy) {
			for (auto gx = 0; gx < gridW; ++gx){
				const int gid1 = gy * (gridW+1) + gx;
				const int gid2 = (gy+1) * (gridW+1) + gx;
				const int gid3 = (gy+1)*(gridW+1)+gx+1;
				const int gid4= gy * (gridW+1) + gx+1;
				if(grid[gid1][0] > 0 && grid[gid2][0] > 0)
					cv::line(img, cv::Point(grid[gid1][0], grid[gid1][1]), cv::Point(grid[gid2][0], grid[gid2][1]), Scalar(255,255,255));
				if(grid[gid2][0] > 0 && grid[gid3][0] > 0)
					cv::line(img, cv::Point(grid[gid2][0], grid[gid2][1]), cv::Point(grid[gid3][0], grid[gid3][1]), Scalar(255,255,255));
				if(grid[gid3][0] > 0 && grid[gid4][0] > 0)
					cv::line(img, cv::Point(grid[gid3][0], grid[gid3][1]), cv::Point(grid[gid4][0], grid[gid4][1]), Scalar(255,255,255));
				if(grid[gid4][0] > 0 && grid[gid1][0] > 0)
					cv::line(img, cv::Point(grid[gid4][0], grid[gid4][1]), cv::Point(grid[gid1][0], grid[gid1][1]), Scalar(255,255,255));
			}
		}
	}

	void GridWarpping::computeSimilarityWeight(const cv::Mat &input, std::vector<double>& saliency) const {
		saliency.resize((size_t)(gridW*gridH));
		for(auto y=0; y<gridH; ++y){
			for(auto x=0; x<gridW; ++x){
				const int gid = gridInd(x,y);
				vector<vector<double> > pixs(3);
				for(auto x1=(int)gridLoc[gridInd(x,y)][0]; x1<gridLoc[gridInd(x+1,y+1)][0]; ++x1){
					for(auto y1=(int)gridLoc[gridInd(x,y)][1]; y1<gridLoc[gridInd(x+1,y+1)][1]; ++y1){
						Vec3b pix = input.at<Vec3b>(y1,x1);
						pixs[0].push_back((double)pix[0] / 255.0);
						pixs[1].push_back((double)pix[1] / 255.0);
						pixs[2].push_back((double)pix[2] / 255.0);
					}
				}
				Vector3d vars(math_util::variance(pixs[0]),math_util::variance(pixs[1]),math_util::variance(pixs[2]));
				saliency[gid] = vars.norm();
			}
		}
	}


	void GridWarpping::warpImageCloseForm(const cv::Mat &input, cv::Mat &output, const vector<Vector2d>& pts1, const vector<Vector2d>& pts2, const int id) const {
		CHECK_EQ(pts1.size(), pts2.size());

		char buffer[1024] = {};

		vector<Vector2d> resGrid(gridLoc.size());
		const int kDataTerm = (int)pts1.size() * 2;
		const int kSimTerm = (gridW-1)*(gridH-1)*8;
		const int kVar = (int)gridLoc.size() * 2;

		vector<Eigen::Triplet<double> > triplets;
		VectorXd B(kDataTerm+kSimTerm);
		//add data constraint
		const double wdata = 1.0;
		const double wsimilarity = 20;
		int cInd = 0;
		for(auto i=0; i<pts2.size(); ++i) {
			if (pts2[i][0] < 0 || pts2[i][1] < 0 || pts2[i][0] >= width - 1 || pts2[i][1] >= height - 1)
				continue;
			Vector4i indRef;
			Vector4d bwRef;
			CHECK_LT(cInd + 1, B.rows());

			getGridIndAndWeight(pts2[i], indRef, bwRef);
			for (auto j = 0; j < 4; ++j) {
				CHECK_LT(indRef[j]*2+1, kVar);
				triplets.push_back(Triplet<double>(cInd, indRef[j] * 2, wdata * bwRef[j]));
				triplets.push_back(Triplet<double>(cInd + 1, indRef[j] * 2 + 1, wdata * bwRef[j]));
			}
			B[cInd] = wdata * pts1[i][0];
			B[cInd + 1] = wdata * pts1[i][1];
			cInd += 2;
		}

		// Mat inputOutput = input.clone();
		// visualizeGrid(gridLoc, inputOutput);
		// for(const auto& pt: pts1)
		// 	cv::circle(inputOutput, cv::Point2d(pt[0], pt[1]), 1, Scalar(0,0,255), 2);
		// sprintf(buffer, "vis_input%05d.jpg", id);
		// imwrite(buffer, inputOutput);

//		vector<double> saliency;
//		computeSimilarityWeight(input, saliency);

		auto getLocalCoord = [](const Vector2d& p1, const Vector2d& p2, const Vector2d& p3){
			Vector2d axis1 = p3 - p2;
			Vector2d axis2(-1*axis1[1], axis1[0]);
			Vector2d v = p1 - p2;
			return Vector2d(v.dot(axis1)/axis1.squaredNorm(), v.dot(axis2)/axis2.squaredNorm());
		};
		{
			//test for local coord
//			Vector2d p1(0,0), p2(0,-1), p3(1,0);
//			Vector2d uv = getLocalCoord(p1,p2,p3);
//			printf("(%.2f,%.2f)\n", uv[0], uv[1]);
		}
		for(auto y=1; y< gridH; ++y) {
			for (auto x = 1; x < gridW; ++x) {
				vector<Vector2i> gids{
						Vector2i(gridInd(x - 1, y), gridInd(x, y - 1)),
						Vector2i(gridInd(x, y - 1), gridInd(x + 1, y)),
						Vector2i(gridInd(x + 1, y), gridInd(x, y + 1)),
						Vector2i(gridInd(x, y + 1), gridInd(x - 1, y))
				};
				const int cgid = gridInd(x, y);
//				printf("-----------------------\n");
				for (const auto &gid: gids) {
					Vector2d refUV = getLocalCoord(gridLoc[cgid], gridLoc[gid[0]], gridLoc[gid[1]]);
//					printf("(%.2f,%.2f),(%.2f,%.2f),(%.2f,%.2f), u:%.2f,v:%.2f\n", gridLoc[cgid][0], gridLoc[cgid][1],
//						   gridLoc[gid[0]][0], gridLoc[gid[0]][1], gridLoc[gid[1]][0], gridLoc[gid[1]][1], refUV[0],
//						   refUV[1]);
					//x coordinate
					triplets.push_back(Triplet<double>(cInd, cgid * 2, wsimilarity));
					triplets.push_back(Triplet<double>(cInd, gid[0]*2, -1 * wsimilarity));
					triplets.push_back(Triplet<double>(cInd, gid[0] * 2, refUV[0] * wsimilarity));
					triplets.push_back(Triplet<double>(cInd, gid[1] * 2, -1 * refUV[0] * wsimilarity));

					triplets.push_back(Triplet<double>(cInd, gid[0] * 2 + 1, -1 * refUV[1] * wsimilarity));
					triplets.push_back(Triplet<double>(cInd, gid[1] * 2 + 1, refUV[1] * wsimilarity));
					B[cInd] = 0;

					//y coordinate
					triplets.push_back(Triplet<double>(cInd + 1, cgid * 2 + 1, wsimilarity));
					triplets.push_back(Triplet<double>(cInd + 1, gid[0] * 2 + 1, -1 * wsimilarity));
					triplets.push_back(Triplet<double>(cInd + 1, gid[0] * 2 + 1, refUV[0] * wsimilarity));
					triplets.push_back(Triplet<double>(cInd + 1, gid[1] * 2 + 1, -1 * refUV[0] * wsimilarity));
					triplets.push_back(Triplet<double>(cInd + 1, gid[0] * 2, refUV[1] * wsimilarity));
					triplets.push_back(Triplet<double>(cInd + 1, gid[1] * 2, -1 * refUV[1] * wsimilarity));
					B[cInd + 1] = 0;
					cInd += 2;
				}
			}
		}

//		const double wregular = 0.1;
//		for(auto x=0; x<=gridW; ++x){
//			for(auto y=0; y<=gridH; ++y){
//				int gid = gridInd(x,y);
//				triplets.push_back(Triplet<double>(cInd, gid*2, wregular));
//				triplets.push_back(Triplet<double>(cInd+1, gid*2+1, wregular));
//				B[cInd] = wregular * gridLoc[gid][0];
//				B[cInd+1] = wregular * gridLoc[gid][1];
//				cInd +=2;
//			}
//		}
		CHECK_LE(cInd, kDataTerm+kSimTerm);
		SparseMatrix<double> A(cInd, kVar);
		A.setFromTriplets(triplets.begin(), triplets.end());

		Eigen::SPQR<SparseMatrix<double> > solver(A);
		VectorXd res = solver.solve(B.block(0,0,cInd,1));
		CHECK_EQ(res.rows(), kVar);

		vector<Vector2d> vars(gridLoc.size());
		for(auto i=0; i<vars.size(); ++i){
			vars[i][0] = res[2*i];
			vars[i][1] = res[2*i+1];
		}

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

		// Mat outputOutput = input.clone();
		// visualizeGrid(vars, outputOutput);
		// for(const auto& pt: pts2)
		// 	cv::circle(outputOutput, cv::Point2d(pt[0], pt[1]), 1, Scalar(0,0,255), 2);
		// sprintf(buffer, "vis_output%05d.jpg", id);
		// imwrite(buffer, outputOutput);
	}

}//namespace substablas
