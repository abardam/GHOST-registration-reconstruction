#pragma once

#include <opencv2/opencv.hpp>

struct SkeletonConstraints{
	float A, B, C, D, E, F, G, H, I, J, K, L;
};


SkeletonConstraints calcSkeletonConstraints(cv::Mat previousRotation, cv::Vec4f previousTranslation, cv::Mat referenceRotation, cv::Vec4f referenceTranslation, float referenceRotationConfidence, float referenceTranslationConfidence);

cv::Mat parametersTransformMatrix(cv::Mat x);

cv::Mat parametersRotationMatrix(cv::Mat x);

void skeleton_constraints_linear(const cv::Mat& prevTransformation, const cv::Mat& refTransformation, float rotationConfidence, float translationConfidence, cv::Mat& out_A, cv::Mat& out_b);
cv::Mat skeleton_constraints_optimize(const cv::Mat& prevTransformation, const cv::Mat& refTransformation, float rotationConfidence, float translationConfidence);
