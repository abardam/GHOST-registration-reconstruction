#include <opencv2\opencv.hpp>
#include "skeletonconstraints.h"

int main(){

	cv::Vec3f previousRotationVector(0,0,0);
	cv::Vec3f referenceRotationVector(.05,0,0);

	cv::Mat previousRotation;
	cv::Mat referenceRotation;

	cv::Rodrigues(previousRotationVector, previousRotation);
	cv::Rodrigues(referenceRotationVector, referenceRotation);

	cv::Vec4f previousTranslation(0, 0, 0, 1);
	cv::Vec4f referenceTranslation(0, 0, 0, 1);

	SkeletonConstraints sc = calcSkeletonConstraints(previousRotation, previousTranslation, referenceRotation, referenceTranslation, 1, 1);

	cv::Mat A = cv::Mat::eye(6, 6, CV_32F);

	A.ptr<float>(3)[3] = sc.G;
	A.ptr<float>(4)[4] = sc.H;
	A.ptr<float>(5)[5] = sc.I;
	A.ptr<float>(3)[4] = sc.A;
	A.ptr<float>(3)[5] = sc.B;
	A.ptr<float>(4)[3] = sc.A;
	A.ptr<float>(5)[3] = sc.B;
	A.ptr<float>(4)[5] = sc.C;
	A.ptr<float>(5)[4] = sc.C;

	//adjust translation by rotation
	A.ptr<float>(0)[3] = previousTranslation(1);
	A.ptr<float>(0)[5] = -previousTranslation(2);
	A.ptr<float>(1)[3] = -previousTranslation(0);
	A.ptr<float>(1)[4] = previousTranslation(2);
	A.ptr<float>(2)[4] = -previousTranslation(1);
	A.ptr<float>(2)[5] = previousTranslation(0);

	float b_data[] = { sc.J, sc.K, sc.L, sc.D, sc.E, sc.F };

	cv::Mat b(6, 1, CV_32F, b_data);

	cv::Mat x;

	cv::solve(A, b, x, cv::DECOMP_CHOLESKY);

	cv::Mat generatedTransform = parametersTransformMatrix(x);
	cv::Mat referenceTransform = cv::Mat::eye(4, 4, CV_32F);
	cv::Mat previousTransform = cv::Mat::eye(4, 4, CV_32F);

	referenceRotation.copyTo(referenceTransform(cv::Range(0, 3), cv::Range(0, 3)));
	cv::Mat(referenceTranslation).copyTo(referenceTransform(cv::Range(0, 4), cv::Range(3, 4)));
	previousRotation.copyTo(previousTransform(cv::Range(0, 3), cv::Range(0, 3)));
	cv::Mat(previousTranslation).copyTo(previousTransform(cv::Range(0, 4), cv::Range(3, 4)));


	std::cout << "x: \n" << x << std::endl;

	std::cout << "reference: \n" << referenceTransform << std::endl;
	std::cout << "previous: \n" << previousTransform << std::endl;

	std::cout << "generated * previous: \n" << generatedTransform*previousTransform << std::endl;

	std::cout << "difference: \n" << referenceTransform - generatedTransform*previousTransform << std::endl;

	int i;
	std::cin >> i;
}