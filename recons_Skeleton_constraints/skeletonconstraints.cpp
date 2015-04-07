#include "skeletonconstraints.h"

SkeletonConstraints calcSkeletonConstraints(cv::Mat previousRotation, cv::Vec4f previousTranslation, cv::Mat referenceRotation, cv::Vec4f referenceTranslation, float referenceRotationConfidence, float referenceTranslationConfidence){

	referenceRotationConfidence /= 3;

	SkeletonConstraints sc;
	sc.A = referenceRotationConfidence * (-previousRotation.ptr<float>(2)[0] * previousRotation.ptr<float>(0)[0] - previousRotation.ptr<float>(2)[1] * previousRotation.ptr<float>(0)[1] - previousRotation.ptr<float>(2)[2] * previousRotation.ptr<float>(0)[2]);
	sc.B = referenceRotationConfidence * (-previousRotation.ptr<float>(2)[0] * previousRotation.ptr<float>(1)[0] - previousRotation.ptr<float>(2)[1] * previousRotation.ptr<float>(1)[1] - previousRotation.ptr<float>(2)[2] * previousRotation.ptr<float>(1)[2]);
	sc.C = referenceRotationConfidence * (-previousRotation.ptr<float>(0)[0] * previousRotation.ptr<float>(1)[0] - previousRotation.ptr<float>(0)[1] * previousRotation.ptr<float>(1)[1] - previousRotation.ptr<float>(0)[2] * previousRotation.ptr<float>(1)[2]);

	sc.D = -referenceRotationConfidence * (-referenceRotation.ptr<float>(0)[0] * previousRotation.ptr<float>(1)[0] + referenceRotation.ptr<float>(1)[0] * previousRotation.ptr<float>(0)[0] - referenceRotation.ptr<float>(0)[1] * previousRotation.ptr<float>(1)[1] + referenceRotation.ptr<float>(1)[1] * previousRotation.ptr<float>(0)[1] - referenceRotation.ptr<float>(0)[2] * previousRotation.ptr<float>(1)[2] + referenceRotation.ptr<float>(1)[2] * previousRotation.ptr<float>(0)[2]);
	sc.E = -referenceRotationConfidence * (-referenceRotation.ptr<float>(1)[0] * previousRotation.ptr<float>(2)[0] + referenceRotation.ptr<float>(2)[0] * previousRotation.ptr<float>(1)[0] - referenceRotation.ptr<float>(1)[1] * previousRotation.ptr<float>(2)[1] + referenceRotation.ptr<float>(2)[1] * previousRotation.ptr<float>(1)[1] - referenceRotation.ptr<float>(1)[2] * previousRotation.ptr<float>(2)[2] + referenceRotation.ptr<float>(2)[2] * previousRotation.ptr<float>(1)[2]);
	sc.F = -referenceRotationConfidence * (+referenceRotation.ptr<float>(0)[0] * previousRotation.ptr<float>(2)[0] - referenceRotation.ptr<float>(2)[0] * previousRotation.ptr<float>(0)[0] + referenceRotation.ptr<float>(0)[1] * previousRotation.ptr<float>(2)[1] - referenceRotation.ptr<float>(2)[1] * previousRotation.ptr<float>(0)[1] + referenceRotation.ptr<float>(0)[2] * previousRotation.ptr<float>(2)[2] - referenceRotation.ptr<float>(2)[2] * previousRotation.ptr<float>(0)[2]);

	sc.G = referenceRotationConfidence * (previousRotation.ptr<float>(1)[0] * previousRotation.ptr<float>(1)[0] + previousRotation.ptr<float>(0)[0] * previousRotation.ptr<float>(0)[0] + previousRotation.ptr<float>(1)[1] * previousRotation.ptr<float>(1)[1] + previousRotation.ptr<float>(0)[1] * previousRotation.ptr<float>(0)[1] + previousRotation.ptr<float>(1)[2] * previousRotation.ptr<float>(1)[2] + previousRotation.ptr<float>(0)[2] * previousRotation.ptr<float>(0)[2]);
	sc.H = referenceRotationConfidence * (previousRotation.ptr<float>(2)[0] * previousRotation.ptr<float>(2)[0] + previousRotation.ptr<float>(1)[0] * previousRotation.ptr<float>(1)[0] + previousRotation.ptr<float>(2)[1] * previousRotation.ptr<float>(2)[1] + previousRotation.ptr<float>(1)[1] * previousRotation.ptr<float>(1)[1] + previousRotation.ptr<float>(2)[2] * previousRotation.ptr<float>(2)[2] + previousRotation.ptr<float>(1)[2] * previousRotation.ptr<float>(1)[2]);
	sc.I = referenceRotationConfidence * (previousRotation.ptr<float>(2)[0] * previousRotation.ptr<float>(2)[0] + previousRotation.ptr<float>(0)[0] * previousRotation.ptr<float>(0)[0] + previousRotation.ptr<float>(2)[1] * previousRotation.ptr<float>(2)[1] + previousRotation.ptr<float>(0)[1] * previousRotation.ptr<float>(0)[1] + previousRotation.ptr<float>(2)[2] * previousRotation.ptr<float>(2)[2] + previousRotation.ptr<float>(0)[2] * previousRotation.ptr<float>(0)[2]);

	sc.J = referenceTranslationConfidence * (referenceTranslation(0) - previousTranslation(0));
	sc.K = referenceTranslationConfidence * (referenceTranslation(1) - previousTranslation(1));
	sc.L = referenceTranslationConfidence * (referenceTranslation(2) - previousTranslation(2));

	return sc;
}

cv::Mat parametersTransformMatrix(cv::Mat x){
	float t_x = x.ptr<float>(0)[0],
		t_y = x.ptr<float>(1)[0],
		t_z = x.ptr<float>(2)[0],
		alpha = x.ptr<float>(3)[0],
		beta = x.ptr<float>(4)[0],
		gamma = x.ptr<float>(5)[0];
	float ret_data[] =
	{ 1, alpha, -gamma, t_x,
	-alpha, 1, beta, t_y,
	gamma, -beta, 1, t_z,
	0, 0, 0, 1 };
	cv::Mat ret = cv::Mat(4, 4, CV_32F, ret_data).clone();
	return ret;
}

cv::Mat parametersRotationMatrix(cv::Mat x){
	return parametersTransformMatrix(x)(cv::Range(0, 3), cv::Range(0, 3));
}

void skeleton_constraints_linear(const cv::Mat& prevTransformation, const cv::Mat& refTransformation, float rotationConfidence, float translationConfidence, cv::Mat& A, cv::Mat& b){
	cv::Range rotateRange[] = { cv::Range(0, 3), cv::Range(0, 3) };
	cv::Mat prevRotation = prevTransformation(rotateRange);
	cv::Vec4f prevTranslation(
		prevTransformation.ptr<float>(0)[3],
		prevTransformation.ptr<float>(1)[3],
		prevTransformation.ptr<float>(2)[3],
		1);
	cv::Mat refRotation = refTransformation(rotateRange);
	cv::Vec4f refTranslation(
		refTransformation.ptr<float>(0)[3],
		refTransformation.ptr<float>(1)[3],
		refTransformation.ptr<float>(2)[3],
		1);

	SkeletonConstraints sc = calcSkeletonConstraints(prevRotation, prevTranslation, refRotation, refTranslation, rotationConfidence, translationConfidence);

	A = cv::Mat::eye(6, 6, CV_32F);

	A.ptr<float>(3)[3] = sc.G;
	A.ptr<float>(4)[4] = sc.H;
	A.ptr<float>(5)[5] = sc.I;
	A.ptr<float>(3)[4] = sc.A;
	A.ptr<float>(3)[5] = sc.B;
	A.ptr<float>(4)[3] = sc.A;
	A.ptr<float>(5)[3] = sc.B;
	A.ptr<float>(4)[5] = sc.C;
	A.ptr<float>(5)[4] = sc.C;

	b = cv::Mat::zeros(6, 1, CV_32F);

	b.ptr<float>(0)[0] = sc.J;
	b.ptr<float>(1)[0] = sc.K;
	b.ptr<float>(2)[0] = sc.L;
	b.ptr<float>(3)[0] = sc.D;
	b.ptr<float>(4)[0] = sc.E;
	b.ptr<float>(5)[0] = sc.F;


	//enzo's edit:

	float tx = prevTransformation.ptr<float>(0)[3];
	float ty = prevTransformation.ptr<float>(1)[3];
	float tz = prevTransformation.ptr<float>(2)[3];
	float dx = refTransformation.ptr<float>(0)[3];
	float dy = refTransformation.ptr<float>(1)[3];
	float dz = refTransformation.ptr<float>(2)[3];

	A.ptr<float>(0)[3] = A.ptr<float>(3)[0] = ty;
	A.ptr<float>(0)[5] = A.ptr<float>(5)[0] = -tz;
	A.ptr<float>(1)[3] = A.ptr<float>(3)[1] = -tx;
	A.ptr<float>(1)[4] = A.ptr<float>(4)[1] = tz;
	A.ptr<float>(2)[4] = A.ptr<float>(4)[2] = -ty;
	A.ptr<float>(2)[5] = A.ptr<float>(5)[2] = tx;

	A.ptr<float>(3)[3] += ty*ty + tx*tx;
	A.ptr<float>(4)[4] += ty*ty + tz*tz;
	A.ptr<float>(5)[5] += tx*tx + tz*tz;

	A.ptr<float>(3)[4] += -tx*tz;
	A.ptr<float>(3)[5] += -ty*tz;
	A.ptr<float>(4)[3] += -tx*tz;
	A.ptr<float>(4)[5] += -tx*ty;
	A.ptr<float>(5)[3] += -ty*tz;
	A.ptr<float>(5)[4] += -tx*ty;

	b.ptr<float>(3)[0] += dx*ty - dy*tx;
	b.ptr<float>(4)[0] += dy*tz - dz*ty;
	b.ptr<float>(5)[0] += dz*tx - dx*tz;

}


cv::Mat skeleton_constraints_optimize(const cv::Mat& prevTransformation, const cv::Mat& refTransformation, float rotationConfidence, float translationConfidence){

	cv::Mat A,b,x;

	skeleton_constraints_linear(prevTransformation, refTransformation, rotationConfidence, translationConfidence, A, b);

	cv::solve(A, b, x, cv::DECOMP_CHOLESKY);

	cv::Mat generatedTransformation = parametersTransformMatrix(x);

	return generatedTransformation;
}