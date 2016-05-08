//
// Created by yanhang on 4/21/16.
//

#include "warping.h"

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


}//namespace substablas
