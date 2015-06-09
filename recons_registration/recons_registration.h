#pragma once

#include <cv_skeleton.h>
#include <recons_voxel.h>

//this version instantly applies transforms and recalculates child transformations immediately after estimation
void estimate_skeleton_and_transform(
	const BodyPartDefinitionVector& bpdv,
	const cv::Mat& source_camera_matrix,
	const cv::Mat& source_camera_pose,
	const cv::Mat& source_color,
	const cv::Mat& source_depth,
	const cv::Mat& target_camera_matrix,
	const cv::Mat& target_camera_pose,
	const cv::Mat& target_color,
	const cv::Mat& target_depth,
	SkeletonNodeHardMap& source_snhmap,
	const SkeletonNodeHardMap& target_snhmap,
	const std::vector<VoxelMatrix>& source_volumes,
	float voxel_size,
	cv::Mat& debug_img = cv::Mat());

//this version is for absolute snh
void estimate_skeleton_and_transform(
	const BodyPartDefinitionVector& bpdv,
	const cv::Mat& source_camera_matrix,
	const cv::Mat& source_camera_pose,
	const cv::Mat& source_color,
	const cv::Mat& source_depth,
	const cv::Mat& target_camera_matrix,
	const cv::Mat& target_camera_pose,
	const cv::Mat& target_color,
	const cv::Mat& target_depth,
	SkeletonNodeAbsoluteVector& source_snav,
	const SkeletonNodeAbsoluteVector& target_snav,
	const std::vector<VoxelMatrix>& source_volumes,
	float voxel_size,
	cv::Mat& debug_img = cv::Mat());

cv::Mat estimate_background_transform(const cv::Mat& source_depth, const cv::Mat& source_rgb, const cv::Mat& source_camera_matrix, const cv::Mat& target_depth, const cv::Mat& target_rgb, const cv::Mat& target_camera_matrix);
cv::Mat estimate_background_transform_multi(const cv::Mat& source_depth, const cv::Mat& source_rgb, const cv::Mat& source_camera_matrix, const std::vector<cv::Mat>& target_depths, const std::vector<cv::Mat>& target_rgbs, const std::vector<cv::Mat>& target_camera_matrixs, const std::vector<cv::Mat>& target_camera_poses);