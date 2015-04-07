#include "point_to_point.h"
#include <opencv2\opencv.hpp>

#define NUMPOINTS 100000

int main(){
	cv::Mat mPoints(4, NUMPOINTS, CV_32F);
	for (int i = 0; i < NUMPOINTS; ++i){
		for (int j = 0; j < 3; ++j){
			mPoints.ptr<float>(j)[i] = std::rand() % 100/100.;
		}
		mPoints.ptr<float>(3)[i] = 1;
	}


	cv::Mat mSourceTransform = cv::Mat::eye(4, 4, CV_32F);
	cv::Mat mTargetTransform = cv::Mat::eye(4, 4, CV_32F);

	mTargetTransform.ptr<float>(0)[3] = 10;

	cv::Vec3f vRotation(.05,0,0);

	cv::Rodrigues(vRotation, mTargetTransform(cv::Range(0, 3), cv::Range(0, 3)));

	cv::Mat C = mSourceTransform * mPoints;
	cv::Mat D = mTargetTransform * mPoints;


	cv::Mat transformDelta = point_to_point_optimize(C, D);

	cv::Mat difference = transformDelta * C - D;
	
	float nrm = 0;
	for (int i = 0; i < NUMPOINTS; ++i){
		nrm += cv::norm(difference(cv::Range(0, 3), cv::Range(i, i + 1)));
	}

	std::cout << nrm*nrm << std::endl;

	int i;
	std::cin >> i;
}