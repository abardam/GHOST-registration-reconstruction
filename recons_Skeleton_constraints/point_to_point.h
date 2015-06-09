#pragma once
#include <opencv2\opencv.hpp>
#include "recons_common.h"

#define OPTICAL_FLOW_ERROR_THRESHOLD 15
#define OPTICAL_FLOW_MIN_MATCHES 40

void point_to_point_linear(const cv::Mat& C, const cv::Mat& D, cv::Mat& out_A, cv::Mat& out_b);
cv::Mat point_to_point_optimize(const cv::Mat& C, const cv::Mat& D);

void point_to_point_registration(
	const cv::Mat& source_pointmat,
	const cv::Mat& source_color,
	const cv::Mat& source_depth,
	const cv::Mat& source_cameramatrix,
	const cv::Mat& source_camerapose_inv,
	const cv::Mat& source_current_transform_delta,
	const cv::Mat& target_color,
	const cv::Mat& target_depth,
	const cv::Mat& target_cameramatrix,
	const cv::Mat& target_camerapose_inv,
	cv::Mat& A, cv::Mat& b,
	bool verbose = false);