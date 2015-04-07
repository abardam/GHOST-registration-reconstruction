#pragma once
#include "opencv2\opencv.hpp"
#include "recons_common.h"

//calculates the normal at each pixel. if mDisplayNormals is provided, returns the normals mapped to color
void calculate_normals(const PointMap& depth_pointmap, const cv::Mat& input_points_2D, cv::Mat& normals, cv::Mat& mDisplayNormals = cv::Mat());

cv::Mat reproject_depth(const cv::Mat& projectedPts, const cv::Mat& depthMat, const cv::Mat& cameraMatrix);

void point_to_plane_linear(const cv::Mat& C, const cv::Mat& D, const cv::Mat& N, cv::Mat& A, cv::Mat& b);

cv::Mat point_to_plane_optimize(const cv::Mat& C, const cv::Mat& D, const cv::Mat& N, float * energy = 0, float weight = 1);

void point_to_plane_registration(
	const cv::Mat& source_pointmat,
	const cv::Mat& source_depth,
	const cv::Mat& source_cameramatrix,
	const cv::Mat& source_camerapose_inv,
	const cv::Mat& target_depth,
	const cv::Mat& target_cameramatrix,
	const cv::Mat& target_camerapose_inv,
	float voxel_size,
	cv::Mat& A, cv::Mat& b,
	bool verbose = false);