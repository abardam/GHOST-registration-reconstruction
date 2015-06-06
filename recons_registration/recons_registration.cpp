#include "recons_registration.h"

#include <recons_voxel_body.h>
#include <point_to_plane.h>
#include <point_to_point.h>
#include <skeletonconstraints.h>

#define RELATIVE_SKELETON 1
#define SKELETON_CONSTRAINT_WEIGHT 1

//this version instantly applies transforms and recalculates child transformations immediately after estimation
//void estimate_skeleton_and_transform(
//	const BodyPartDefinitionVector& bpdv,
//	const FrameData& source_framedata,
//	const FrameData& target_framedata,
//	SkeletonNodeHardMap& source_snhmap,
//	const SkeletonNodeHardMap& target_snhmap,
//	const std::vector<VoxelMatrix>& source_volumes,
//	float voxel_size,
//	cv::Mat& debug_img){
//
//	bool rel_skel = RELATIVE_SKELETON;
//
//	std::vector<cv::Mat> bodypart_transforms(bpdv.size());
//
//	static int debug_frame = 0;
//
//	std::stringstream ss;
//
//	for (int i = 0; i < bpdv.size(); ++i){
//
//		cv::Mat finalTransformDelta = cv::Mat::eye(4, 4, CV_32F);
//
//		const SkeletonNodeHard * source_node = source_snhmap.find(bpdv[i].mNode1Name)->second;
//		const SkeletonNodeHard * target_node = target_snhmap.find(bpdv[i].mNode1Name)->second;
//
//		cv::Mat source_bodypart_transform = get_bodypart_transform(bpdv[i], source_snhmap, source_framedata.mmCameraPose);
//		cv::Mat target_bodypart_transform = get_bodypart_transform(bpdv[i], target_snhmap, target_framedata.mmCameraPose);
//
//		cv::Mat source_bodypart_transform_inv = source_bodypart_transform.inv();
//		cv::Mat target_bodypart_transform_inv = target_bodypart_transform.inv();
//
//		cv::Mat source_screen_transform = source_bodypart_transform * get_voxel_transform(source_volumes[i].width, source_volumes[i].height, source_volumes[i].depth, voxel_size);
//
//		cv::Mat source_bodypart_pointmat;
//		{
//			std::vector<cv::Vec4f> source_points;
//			for (int j = 0; j < source_volumes[i].voxel_data.cols; ++j){
//				if (source_volumes[i].voxel_data.ptr<cv::Vec4b>()[j](3) == 0xff){
//					source_points.push_back(cv::Vec4f(
//						source_volumes[i].voxel_coords.ptr<float>(0)[j],
//						source_volumes[i].voxel_coords.ptr<float>(1)[j],
//						source_volumes[i].voxel_coords.ptr<float>(2)[j],
//						1
//						));
//				}
//			}
//
//			cv::Mat source_bodypart_pointmat_r(1, source_points.size(), CV_32FC4, source_points.data());
//			source_bodypart_pointmat = source_bodypart_pointmat_r.reshape(1, source_bodypart_pointmat_r.cols).t();
//
//
//			if (source_points.size() > 0){
//				//convert to screen coordinates
//				source_bodypart_pointmat = source_screen_transform * source_bodypart_pointmat;
//			}
//		}
//
//
//		for (int iter = 0; iter<50; ++iter){ //straight up lets just do 50 iterations
//
//			ss.str("");
//			cv::FileStorage fs;
//			ss << "LDTP-debug/debug-frame" << debug_frame << "-bp" << i << ".yml";
//			fs.open(ss.str(), cv::FileStorage::WRITE);
//
//
//			cv::Mat A(6, 6, CV_64F, cv::Scalar(0));
//			cv::Mat b(6, 1, CV_64F, cv::Scalar(0));
//
//
//			//cv::Mat source_parent_bodypart_transform_inv = (source_bodypart_transform * source_node->mTransformation.inv()).inv();
//			//cv::Mat target_parent_bodypart_transform_inv = (target_bodypart_transform * target_node->mTransformation.inv()).inv();
//
//			//point to plane
//			if (!source_bodypart_pointmat.empty())
//			{
//				cv::Mat _A, _b;
//
//				if (false){
//					point_to_plane_registration(source_bodypart_pointmat, source_framedata.mmDepth, source_framedata.mmCameraMatrix, source_bodypart_transform_inv,
//						finalTransformDelta,
//						target_framedata.mmDepth, target_framedata.mmCameraMatrix,
//						target_bodypart_transform_inv, voxel_size, _A, _b, true);
//				}
//				else{
//					//try multiplying the inverse camera pose instead of the body part transform
//					point_to_plane_registration(source_bodypart_pointmat, source_framedata.mmDepth, source_framedata.mmCameraMatrix, (source_framedata.mmCameraPose).inv(),
//						finalTransformDelta,
//						target_framedata.mmDepth, target_framedata.mmCameraMatrix,
//						target_framedata.mmCameraPose.inv(), voxel_size, _A, _b, true);
//				}
//				//try multiplying the inverse body part transform of the PARENT //update: not very good...
//
//				//point_to_plane_registration(source_bodypart_pointmat, source_framedata.mmDepth, source_framedata.mmCameraMatrix, source_parent_bodypart_transform_inv,
//				//	target_framedata.mmDepth, target_framedata.mmCameraMatrix,
//				//	target_parent_bodypart_transform_inv, voxel_size, _A, _b, true);
//
//				cv::Mat A64, b64;
//				_A.convertTo(A64, CV_64F);
//				_b.convertTo(b64, CV_64F);
//
//				cv::add(A, A64, A);
//				cv::add(b, b64, b);
//
//				//debug
//
//				cv::Mat x;
//
//				cv::solve(A64, b64, x, cv::DECOMP_CHOLESKY);
//				float energy = cv::norm(A*x - b);
//				cv::Mat transformDelta = cv::Mat::eye(4, 4, CV_32F);
//
//				transformDelta.ptr<float>(0)[1] = x.ptr<double>(3)[0];
//				transformDelta.ptr<float>(1)[0] = -x.ptr<double>(3)[0];
//				transformDelta.ptr<float>(0)[2] = -x.ptr<double>(5)[0];
//				transformDelta.ptr<float>(2)[0] = x.ptr<double>(5)[0];
//				transformDelta.ptr<float>(1)[2] = x.ptr<double>(4)[0];
//				transformDelta.ptr<float>(2)[1] = -x.ptr<double>(4)[0];
//				transformDelta.ptr<float>(0)[3] = x.ptr<double>(0)[0];
//				transformDelta.ptr<float>(1)[3] = x.ptr<double>(1)[0];
//				transformDelta.ptr<float>(2)[3] = x.ptr<double>(2)[0];
//
//				std::cout << "point to plane transform: \n" << transformDelta << std::endl;
//
//				fs << "Point_to_plane" << "{"
//					<< "A" << A64
//					<< "b" << b64
//					<< "x" << x
//					<< "transform" << transformDelta
//					<< "}";
//
//			}
//
//			//point to point
//			//if (!source_bodypart_pointmat.empty())
//			//{
//			//	cv::Mat _A, _b;
//			//
//			//	if (false){
//			//		point_to_point_registration(source_bodypart_pointmat, source_framedata.mmColor, source_framedata.mmDepth, source_framedata.mmCameraMatrix, source_bodypart_transform_inv,
//			//			finalTransformDelta,
//			//			target_framedata.mmColor, target_framedata.mmDepth, target_framedata.mmCameraMatrix, target_bodypart_transform_inv, _A, _b, true);
//			//	}
//			//	else{
//			//		point_to_point_registration(source_bodypart_pointmat, source_framedata.mmColor, source_framedata.mmDepth, source_framedata.mmCameraMatrix, source_framedata.mmCameraPose.inv(),
//			//			finalTransformDelta,
//			//			target_framedata.mmColor, target_framedata.mmDepth, target_framedata.mmCameraMatrix, target_framedata.mmCameraPose.inv(), _A, _b, true);
//			//	}
//			//	//point_to_point_registration(source_bodypart_pointmat, source_framedata.mmColor, source_framedata.mmDepth, source_framedata.mmCameraMatrix, source_parent_bodypart_transform_inv,
//			//	//	target_framedata.mmColor, target_framedata.mmDepth, target_framedata.mmCameraMatrix, target_parent_bodypart_transform_inv, _A, _b, true);
//			//
//			//	cv::Mat A64, b64;
//			//	_A.convertTo(A64, CV_64F);
//			//	_b.convertTo(b64, CV_64F);
//			//
//			//	cv::add(A, A64, A);
//			//	cv::add(b, b64, b);
//			//
//			//	//debug
//			//
//			//	cv::Mat x;
//			//
//			//	cv::solve(A64, b64, x, cv::DECOMP_CHOLESKY);
//			//	float energy = cv::norm(A*x - b);
//			//	cv::Mat transformDelta = cv::Mat::eye(4, 4, CV_32F);
//			//
//			//	transformDelta.ptr<float>(0)[1] = x.ptr<double>(3)[0];
//			//	transformDelta.ptr<float>(1)[0] = -x.ptr<double>(3)[0];
//			//	transformDelta.ptr<float>(0)[2] = -x.ptr<double>(5)[0];
//			//	transformDelta.ptr<float>(2)[0] = x.ptr<double>(5)[0];
//			//	transformDelta.ptr<float>(1)[2] = x.ptr<double>(4)[0];
//			//	transformDelta.ptr<float>(2)[1] = -x.ptr<double>(4)[0];
//			//	transformDelta.ptr<float>(0)[3] = x.ptr<double>(0)[0];
//			//	transformDelta.ptr<float>(1)[3] = x.ptr<double>(1)[0];
//			//	transformDelta.ptr<float>(2)[3] = x.ptr<double>(2)[0];
//			//	std::cout << "point to point transform: \n" << transformDelta << std::endl;
//			//
//			//	fs << "Point_to_point" << "{"
//			//		<< "A" << A64
//			//		<< "b" << b64
//			//		<< "x" << x
//			//		<< "transform" << transformDelta
//			//		<< "}";
//			//}
//
//			//skeleton
//			{
//				cv::Mat _A, _b;
//
//				if (rel_skel){
//					skeleton_constraints_linear(finalTransformDelta * source_node->mTransformation, target_node->mTransformation, 1, 1, _A, _b);
//				}
//				else{
//					skeleton_constraints_linear(source_bodypart_transform, target_bodypart_transform, 1, 1, _A, _b);
//				}
//				cv::Mat A64, b64;
//				_A.convertTo(A64, CV_64F);
//				_b.convertTo(b64, CV_64F);
//
//				cv::add(A, target_node->confidence * SKELETON_CONSTRAINT_WEIGHT * A64, A);
//				cv::add(b, target_node->confidence * SKELETON_CONSTRAINT_WEIGHT * b64, b);
//
//				//debug
//
//				cv::Mat x;
//
//				cv::solve(A64, b64, x, cv::DECOMP_CHOLESKY);
//				float energy = cv::norm(A*x - b);
//				cv::Mat transformDelta = cv::Mat::eye(4, 4, CV_32F);
//
//				transformDelta.ptr<float>(0)[1] = x.ptr<double>(3)[0];
//				transformDelta.ptr<float>(1)[0] = -x.ptr<double>(3)[0];
//				transformDelta.ptr<float>(0)[2] = -x.ptr<double>(5)[0];
//				transformDelta.ptr<float>(2)[0] = x.ptr<double>(5)[0];
//				transformDelta.ptr<float>(1)[2] = x.ptr<double>(4)[0];
//				transformDelta.ptr<float>(2)[1] = -x.ptr<double>(4)[0];
//				transformDelta.ptr<float>(0)[3] = x.ptr<double>(0)[0];
//				transformDelta.ptr<float>(1)[3] = x.ptr<double>(1)[0];
//				transformDelta.ptr<float>(2)[3] = x.ptr<double>(2)[0];
//				std::cout << "skeleton transform: \n" << transformDelta << std::endl;
//
//				fs << "skeleton" << "{"
//					<< "A" << A64
//					<< "b" << b64
//					<< "x" << x
//					<< "transform" << transformDelta
//					<< "}";
//			}
//
//
//			cv::Mat x;
//
//			cv::solve(A, b, x, cv::DECOMP_CHOLESKY);
//			float energy = cv::norm(A*x - b);
//			cv::Mat transformDelta = cv::Mat::eye(4, 4, CV_32F);
//
//			transformDelta.ptr<float>(0)[1] = x.ptr<double>(3)[0];
//			transformDelta.ptr<float>(1)[0] = -x.ptr<double>(3)[0];
//			transformDelta.ptr<float>(0)[2] = -x.ptr<double>(5)[0];
//			transformDelta.ptr<float>(2)[0] = x.ptr<double>(5)[0];
//			transformDelta.ptr<float>(1)[2] = x.ptr<double>(4)[0];
//			transformDelta.ptr<float>(2)[1] = -x.ptr<double>(4)[0];
//			transformDelta.ptr<float>(0)[3] = x.ptr<double>(0)[0];
//			transformDelta.ptr<float>(1)[3] = x.ptr<double>(1)[0];
//			transformDelta.ptr<float>(2)[3] = x.ptr<double>(2)[0];
//
//			{
//				//extract scale and correct it?
//				cv::Vec3f scale(
//					1 / cv::norm(transformDelta(cv::Range(0, 1), cv::Range(0, 3))),
//					1 / cv::norm(transformDelta(cv::Range(1, 2), cv::Range(0, 3))),
//					1 / cv::norm(transformDelta(cv::Range(2, 3), cv::Range(0, 3)))
//					);
//
//				cv::Mat rot = cv::Mat::diag(cv::Mat(scale)) * transformDelta(cv::Range(0, 3), cv::Range(0, 3));
//				rot.copyTo(transformDelta(cv::Range(0, 3), cv::Range(0, 3)));
//			}
//
//			finalTransformDelta = transformDelta * finalTransformDelta;
//
//			std::cout << "final transform: \n" << transformDelta << std::endl;
//
//
//			fs << "Final" << "{"
//				<< "A" << A
//				<< "b" << b
//				<< "x" << x
//				<< "transform" << transformDelta
//				<< "}";
//
//			fs.release();
//
//			if (!debug_img.empty()){
//
//				cv::Mat debug_img_volumes = debug_img.clone();
//
//				for (int i = 0; i < bpdv.size(); ++i){
//					cv::Vec3b color(bpdv[i].mColor[2] * 255, bpdv[i].mColor[1] * 255, bpdv[i].mColor[0] * 255);
//					voxel_draw_volume(debug_img_volumes, color, source_framedata.mmCameraPose * finalTransformDelta * get_bodypart_transform(bpdv[i], source_snhmap, cv::Mat::eye(4, 4, CV_32F)), source_framedata.mmCameraMatrix, &source_volumes[i], voxel_size);
//				}
//
//				ss.str("");
//				ss << "LDTP-debug/debug-frame" << debug_frame << "-bp" << i << ".png";
//				cv::imwrite(ss.str(), debug_img_volumes);
//
//				cv::imshow("debug volumes", debug_img_volumes);
//				cv::waitKey(10);
//
//
//			}
//
//			//break iteration
//			cv::Mat cmp_mat = cv::Mat::eye(4, 4, CV_32F) - transformDelta;
//			cv::Mat i_cmp = cmp_mat(cv::Range(0, 3), cv::Range(0, 1));
//			cv::Mat j_cmp = cmp_mat(cv::Range(0, 3), cv::Range(1, 2));
//			cv::Mat k_cmp = cmp_mat(cv::Range(0, 3), cv::Range(2, 3));
//			cv::Mat t_cmp = cmp_mat(cv::Range(0, 3), cv::Range(3, 4));
//			float diff = cv::norm(i_cmp) + cv::norm(j_cmp) + cv::norm(k_cmp) + cv::norm(t_cmp);
//			if (diff < 0.001) break;
//		}
//
//		SkeletonNodeHard * node = source_snhmap.find(bpdv[i].mNode1Name)->second;
//		SkeletonNodeHard * node_parent = source_snhmap.find(node->mParentName)->second;
//
//		if (rel_skel){
//			node->mTransformation = finalTransformDelta * node->mTransformation;
//			cv_draw_and_build_skeleton(node, node_parent->mTempTransformation, source_framedata.mmCameraMatrix, source_framedata.mmCameraPose, &source_snhmap);
//		}
//		else{
//			node->mTempTransformation = finalTransformDelta * node->mTempTransformation;
//			node->mTransformation = node->mTempTransformation * node_parent->mTempTransformation.inv();
//		}
//
//
//	}
//
//	++debug_frame;
//}
//

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
	cv::Mat& debug_img){

	bool rel_skel = RELATIVE_SKELETON;

	std::vector<cv::Mat> bodypart_transforms(bpdv.size());

	static int debug_frame = 0;

	std::stringstream ss;

	for (int i = 0; i < bpdv.size(); ++i){

		cv::Mat finalTransformDelta = cv::Mat::eye(4, 4, CV_32F);

		int snh_i = -1;
		for (int j = 0; j < source_snav.size(); ++j){
			if (bpdv[i].mNode1Name == source_snav[j].mName){
				snh_i = j;
				break;
			}
		}

		if (snh_i == -1){
			std::cout << "Something went wrong: skeleton node not found\n";
			return;
		}

		SkeletonNodeHard& source_node = source_snav[snh_i];
		const SkeletonNodeHard& target_node = target_snav[snh_i];

		const cv::Mat& source_bodypart_transform = source_node.mTransformation;
		const cv::Mat& target_bodypart_transform = target_node.mTransformation;

		cv::Mat source_bodypart_transform_inv = source_bodypart_transform.inv();

		cv::Mat source_screen_transform = source_bodypart_transform * get_voxel_transform(source_volumes[i].width, source_volumes[i].height, source_volumes[i].depth, voxel_size);

		cv::Mat source_bodypart_pointmat;
		{
			std::vector<cv::Vec4f> source_points;
			for (int j = 0; j < source_volumes[i].voxel_data.cols; ++j){
				//if (source_volumes[i].voxel_data.ptr<cv::Vec4b>()[j](3) == 0xff){
				if (true){
					source_points.push_back(cv::Vec4f(
						source_volumes[i].voxel_coords.ptr<float>(0)[j],
						source_volumes[i].voxel_coords.ptr<float>(1)[j],
						source_volumes[i].voxel_coords.ptr<float>(2)[j],
						1
						));
				}
			}

			cv::Mat source_bodypart_pointmat_r(1, source_points.size(), CV_32FC4, source_points.data());
			source_bodypart_pointmat = source_bodypart_pointmat_r.reshape(1, source_bodypart_pointmat_r.cols).t();


			if (source_points.size() > 0){
				//convert to screen coordinates
				source_bodypart_pointmat = source_screen_transform * source_bodypart_pointmat;
			}
		}


		for (int iter = 0; iter<50; ++iter){ //straight up lets just do 50 iterations

			ss.str("");
			cv::FileStorage fs;
			ss << "LDTP-debug/debug-frame" << debug_frame << "-bp" << i << ".yml";
			fs.open(ss.str(), cv::FileStorage::WRITE);


			cv::Mat A(6, 6, CV_64F, cv::Scalar(0));
			cv::Mat b(6, 1, CV_64F, cv::Scalar(0));


			//cv::Mat source_parent_bodypart_transform_inv = (source_bodypart_transform * source_node->mTransformation.inv()).inv();
			//cv::Mat target_parent_bodypart_transform_inv = (target_bodypart_transform * target_node->mTransformation.inv()).inv();

			//point to plane
			if (!source_bodypart_pointmat.empty())
			{
				cv::Mat _A, _b;

				if (false){
				}
				else{
					//try multiplying the inverse camera pose instead of the body part transform
					point_to_plane_registration(source_bodypart_pointmat, source_depth, source_camera_matrix, (source_camera_pose).inv(),
						finalTransformDelta,
						target_depth, target_camera_matrix,
						target_camera_pose.inv(), voxel_size, _A, _b, true);
				}
				//try multiplying the inverse body part transform of the PARENT //update: not very good...

				//point_to_plane_registration(source_bodypart_pointmat, source_framedata.mmDepth, source_framedata.mmCameraMatrix, source_parent_bodypart_transform_inv,
				//	target_framedata.mmDepth, target_framedata.mmCameraMatrix,
				//	target_parent_bodypart_transform_inv, voxel_size, _A, _b, true);

				cv::Mat A64, b64;
				_A.convertTo(A64, CV_64F);
				_b.convertTo(b64, CV_64F);

				cv::add(A, A64, A);
				cv::add(b, b64, b);

				//debug

				cv::Mat x;

				cv::solve(A64, b64, x, cv::DECOMP_CHOLESKY);
				float energy = cv::norm(A*x - b);
				cv::Mat transformDelta = cv::Mat::eye(4, 4, CV_32F);

				transformDelta.ptr<float>(0)[1] = x.ptr<double>(3)[0];
				transformDelta.ptr<float>(1)[0] = -x.ptr<double>(3)[0];
				transformDelta.ptr<float>(0)[2] = -x.ptr<double>(5)[0];
				transformDelta.ptr<float>(2)[0] = x.ptr<double>(5)[0];
				transformDelta.ptr<float>(1)[2] = x.ptr<double>(4)[0];
				transformDelta.ptr<float>(2)[1] = -x.ptr<double>(4)[0];
				transformDelta.ptr<float>(0)[3] = x.ptr<double>(0)[0];
				transformDelta.ptr<float>(1)[3] = x.ptr<double>(1)[0];
				transformDelta.ptr<float>(2)[3] = x.ptr<double>(2)[0];

				std::cout << "point to plane transform: \n" << transformDelta << std::endl;

				fs << "Point_to_plane" << "{"
					<< "A" << A64
					<< "b" << b64
					<< "x" << x
					<< "transform" << transformDelta
					<< "}";

			}

			//point to point
			if (!source_bodypart_pointmat.empty())
			{
				cv::Mat _A, _b;
			
				if (false){
					//point_to_point_registration(source_bodypart_pointmat, source_color, source_depth, source_camera_matrix, source_bodypart_transform_inv,
					//	finalTransformDelta,
					//	target_color, target_depth, target_camera_matrix, target_bodypart_transform_inv, _A, _b, true);
				}
				else{
					point_to_point_registration(source_bodypart_pointmat, source_color, source_depth, source_camera_matrix, source_camera_pose.inv(),
						finalTransformDelta,
						target_color, target_depth, target_camera_matrix, target_camera_pose.inv(), _A, _b, true);
				}
				//point_to_point_registration(source_bodypart_pointmat, source_framedata.mmColor, source_framedata.mmDepth, source_framedata.mmCameraMatrix, source_parent_bodypart_transform_inv,
				//	target_framedata.mmColor, target_framedata.mmDepth, target_framedata.mmCameraMatrix, target_parent_bodypart_transform_inv, _A, _b, true);
			
				cv::Mat A64, b64;
				_A.convertTo(A64, CV_64F);
				_b.convertTo(b64, CV_64F);
			
				cv::add(A, A64, A);
				cv::add(b, b64, b);
			
				//debug
			
				cv::Mat x;
			
				cv::solve(A64, b64, x, cv::DECOMP_CHOLESKY);
				float energy = cv::norm(A*x - b);
				cv::Mat transformDelta = cv::Mat::eye(4, 4, CV_32F);
			
				transformDelta.ptr<float>(0)[1] = x.ptr<double>(3)[0];
				transformDelta.ptr<float>(1)[0] = -x.ptr<double>(3)[0];
				transformDelta.ptr<float>(0)[2] = -x.ptr<double>(5)[0];
				transformDelta.ptr<float>(2)[0] = x.ptr<double>(5)[0];
				transformDelta.ptr<float>(1)[2] = x.ptr<double>(4)[0];
				transformDelta.ptr<float>(2)[1] = -x.ptr<double>(4)[0];
				transformDelta.ptr<float>(0)[3] = x.ptr<double>(0)[0];
				transformDelta.ptr<float>(1)[3] = x.ptr<double>(1)[0];
				transformDelta.ptr<float>(2)[3] = x.ptr<double>(2)[0];
				std::cout << "point to point transform: \n" << transformDelta << std::endl;
			
				fs << "Point_to_point" << "{"
					<< "A" << A64
					<< "b" << b64
					<< "x" << x
					<< "transform" << transformDelta
					<< "}";
			}

			//skeleton
			{
				cv::Mat _A, _b;

				if (rel_skel){
					skeleton_constraints_linear(finalTransformDelta * source_bodypart_transform, target_bodypart_transform, 1, 1, _A, _b);
				}
				else{
					skeleton_constraints_linear(source_bodypart_transform, target_bodypart_transform, 1, 1, _A, _b);
				}
				cv::Mat A64, b64;
				_A.convertTo(A64, CV_64F);
				_b.convertTo(b64, CV_64F);

				cv::add(A, target_node.confidence * SKELETON_CONSTRAINT_WEIGHT * A64, A);
				cv::add(b, target_node.confidence * SKELETON_CONSTRAINT_WEIGHT * b64, b);

				//debug

				cv::Mat x;

				cv::solve(A64, b64, x, cv::DECOMP_CHOLESKY);
				float energy = cv::norm(A*x - b);
				cv::Mat transformDelta = cv::Mat::eye(4, 4, CV_32F);

				transformDelta.ptr<float>(0)[1] = x.ptr<double>(3)[0];
				transformDelta.ptr<float>(1)[0] = -x.ptr<double>(3)[0];
				transformDelta.ptr<float>(0)[2] = -x.ptr<double>(5)[0];
				transformDelta.ptr<float>(2)[0] = x.ptr<double>(5)[0];
				transformDelta.ptr<float>(1)[2] = x.ptr<double>(4)[0];
				transformDelta.ptr<float>(2)[1] = -x.ptr<double>(4)[0];
				transformDelta.ptr<float>(0)[3] = x.ptr<double>(0)[0];
				transformDelta.ptr<float>(1)[3] = x.ptr<double>(1)[0];
				transformDelta.ptr<float>(2)[3] = x.ptr<double>(2)[0];
				std::cout << "skeleton transform: \n" << transformDelta << std::endl;

				fs << "skeleton" << "{"
					<< "A" << A64
					<< "b" << b64
					<< "x" << x
					<< "transform" << transformDelta
					<< "}";
			}


			cv::Mat x;

			cv::solve(A, b, x, cv::DECOMP_CHOLESKY);
			float energy = cv::norm(A*x - b);
			cv::Mat transformDelta = cv::Mat::eye(4, 4, CV_32F);

			transformDelta.ptr<float>(0)[1] = x.ptr<double>(3)[0];
			transformDelta.ptr<float>(1)[0] = -x.ptr<double>(3)[0];
			transformDelta.ptr<float>(0)[2] = -x.ptr<double>(5)[0];
			transformDelta.ptr<float>(2)[0] = x.ptr<double>(5)[0];
			transformDelta.ptr<float>(1)[2] = x.ptr<double>(4)[0];
			transformDelta.ptr<float>(2)[1] = -x.ptr<double>(4)[0];
			transformDelta.ptr<float>(0)[3] = x.ptr<double>(0)[0];
			transformDelta.ptr<float>(1)[3] = x.ptr<double>(1)[0];
			transformDelta.ptr<float>(2)[3] = x.ptr<double>(2)[0];

			{
				//extract scale and correct it?
				cv::Vec3f scale(
					1 / cv::norm(transformDelta(cv::Range(0, 1), cv::Range(0, 3))),
					1 / cv::norm(transformDelta(cv::Range(1, 2), cv::Range(0, 3))),
					1 / cv::norm(transformDelta(cv::Range(2, 3), cv::Range(0, 3)))
					);

				cv::Mat rot = cv::Mat::diag(cv::Mat(scale)) * transformDelta(cv::Range(0, 3), cv::Range(0, 3));
				rot.copyTo(transformDelta(cv::Range(0, 3), cv::Range(0, 3)));
			}

			finalTransformDelta = transformDelta * finalTransformDelta;

			std::cout << "final transform: \n" << transformDelta << std::endl;


			fs << "Final" << "{"
				<< "A" << A
				<< "b" << b
				<< "x" << x
				<< "transform" << transformDelta
				<< "}";

			fs.release();

			if (!debug_img.empty()){

				cv::Mat debug_img_volumes = debug_img.clone();

				for (int i = 0; i < bpdv.size(); ++i){
					cv::Vec3b color(bpdv[i].mColor[2] * 255, bpdv[i].mColor[1] * 255, bpdv[i].mColor[0] * 255);
					voxel_draw_volume(debug_img_volumes, color, source_camera_pose * finalTransformDelta * get_bodypart_transform(bpdv[i], source_snav, cv::Mat::eye(4, 4, CV_32F)),
						source_camera_matrix, &source_volumes[i], voxel_size);
				}

				ss.str("");
				ss << "LDTP-debug/debug-frame" << debug_frame << "-bp" << i << ".png";
				cv::imwrite(ss.str(), debug_img_volumes);

				cv::imshow("debug volumes", debug_img_volumes);
				cv::waitKey(10);


			}

			//break iteration
			cv::Mat cmp_mat = cv::Mat::eye(4, 4, CV_32F) - transformDelta;
			cv::Mat i_cmp = cmp_mat(cv::Range(0, 3), cv::Range(0, 1));
			cv::Mat j_cmp = cmp_mat(cv::Range(0, 3), cv::Range(1, 2));
			cv::Mat k_cmp = cmp_mat(cv::Range(0, 3), cv::Range(2, 3));
			cv::Mat t_cmp = cmp_mat(cv::Range(0, 3), cv::Range(3, 4));
			float diff = cv::norm(i_cmp) + cv::norm(j_cmp) + cv::norm(k_cmp) + cv::norm(t_cmp);
			if (diff < 0.001) break;
		}

		source_node.mTransformation = finalTransformDelta * source_node.mTransformation;

	}

	++debug_frame;
}

cv::Mat estimate_background_transform(const cv::Mat& source_depth, const cv::Mat& source_rgb, const cv::Mat& source_camera_matrix, const cv::Mat& target_depth, const cv::Mat& target_rgb, const cv::Mat& target_camera_matrix){
	cv::Mat depth_points;
	{
		PointMap point_map(source_depth.cols, source_depth.rows);
		read_depth_image(source_depth, source_camera_matrix, point_map);
		depth_points = cv::Mat(4, point_map.mvPoints.size(), CV_32F);
		read_points_pointcloud(point_map, depth_points);
	}

	cv::Mat final_transform_delta = cv::Mat::eye(4, 4, CV_32F);

	for (int iter = 0; iter<50; ++iter){
		cv::Mat A(6, 6, CV_32F, cv::Scalar(0)), b(6, 1, CV_32F, cv::Scalar(0));

		cv::Mat _A, _b;
		point_to_plane_registration(depth_points, source_depth, source_camera_matrix, cv::Mat::eye(4, 4, CV_32F), final_transform_delta, target_depth, target_camera_matrix, cv::Mat::eye(4, 4, CV_32F), 0.1, _A, _b);

		cv::add(A, _A, A);
		cv::add(b, _b, b);


		point_to_point_registration(depth_points, source_rgb, source_depth, source_camera_matrix, cv::Mat::eye(4, 4, CV_32F), final_transform_delta, target_rgb, target_depth, target_camera_matrix, cv::Mat::eye(4, 4, CV_32F), _A, _b);

		cv::add(A, _A, A);
		cv::add(b, _b, b);

		cv::Mat x;

		cv::solve(A, b, x, cv::DECOMP_CHOLESKY);
		float energy = cv::norm(A*x - b);
		cv::Mat transformDelta = cv::Mat::eye(4, 4, CV_32F);

		transformDelta.ptr<float>(0)[1] = x.ptr<float>(3)[0];
		transformDelta.ptr<float>(1)[0] = -x.ptr<float>(3)[0];
		transformDelta.ptr<float>(0)[2] = -x.ptr<float>(5)[0];
		transformDelta.ptr<float>(2)[0] = x.ptr<float>(5)[0];
		transformDelta.ptr<float>(1)[2] = x.ptr<float>(4)[0];
		transformDelta.ptr<float>(2)[1] = -x.ptr<float>(4)[0];
		transformDelta.ptr<float>(0)[3] = x.ptr<float>(0)[0];
		transformDelta.ptr<float>(1)[3] = x.ptr<float>(1)[0];
		transformDelta.ptr<float>(2)[3] = x.ptr<float>(2)[0];

		final_transform_delta = transformDelta * final_transform_delta;

		cv::Mat cmp = transformDelta - cv::Mat::eye(4, 4, CV_32F);
		float diff = cv::norm(cmp.col(0)) + cv::norm(cmp.col(1)) + cv::norm(cmp.col(2)) + cv::norm(cmp.col(3));
		if (diff < 0.001) break;
	}

	return final_transform_delta;
}


cv::Mat estimate_background_transform_multi(const cv::Mat& source_depth, const cv::Mat& source_rgb, const cv::Mat& source_camera_matrix, const std::vector<cv::Mat>& target_depths, const std::vector<cv::Mat>& target_rgbs, const std::vector<cv::Mat>& target_camera_matrixs, const std::vector<cv::Mat>& target_camera_poses){
	cv::Mat depth_points;
	{
		PointMap point_map(source_depth.cols, source_depth.rows);
		read_depth_image(source_depth, source_camera_matrix, point_map);
		depth_points = cv::Mat(4, point_map.mvPoints.size(), CV_32F);
		read_points_pointcloud(point_map, depth_points);
	}

	cv::Mat final_transform_delta = cv::Mat::eye(4, 4, CV_32F);

	if (target_depths.size() != target_rgbs.size() || target_rgbs.size() != target_camera_matrixs.size() || target_camera_matrixs.size() != target_camera_poses.size()){
		std::cout << "Wrong target sizes: estimate_background_transform_multi\n";
		return cv::Mat();
	}

	for (int iter = 0; iter < 50; ++iter){
		cv::Mat A(6, 6, CV_32F, cv::Scalar(0)), b(6, 1, CV_32F, cv::Scalar(0));

		for (int i = 0; i < target_depths.size(); ++i){

			cv::Mat _A, _b;
			point_to_plane_registration(depth_points, source_depth, source_camera_matrix, cv::Mat::eye(4, 4, CV_32F), final_transform_delta, target_depths[i], target_camera_matrixs[i], target_camera_poses[i].inv(), 0.1, _A, _b);

			cv::add(A, _A, A);
			cv::add(b, _b, b);

			if (!source_rgb.empty() && !target_rgbs[i].empty()){
				point_to_point_registration(depth_points, source_rgb, source_depth, source_camera_matrix, cv::Mat::eye(4, 4, CV_32F), final_transform_delta, target_rgbs[i], target_depths[i], target_camera_matrixs[i], target_camera_poses[i].inv(), _A, _b);

				cv::add(A, _A, A);
				cv::add(b, _b, b);
			}

		}

		cv::Mat x;

		cv::solve(A, b, x, cv::DECOMP_CHOLESKY);
		float energy = cv::norm(A*x - b);
		cv::Mat transformDelta = cv::Mat::eye(4, 4, CV_32F);

		transformDelta.ptr<float>(0)[1] = x.ptr<float>(3)[0];
		transformDelta.ptr<float>(1)[0] = -x.ptr<float>(3)[0];
		transformDelta.ptr<float>(0)[2] = -x.ptr<float>(5)[0];
		transformDelta.ptr<float>(2)[0] = x.ptr<float>(5)[0];
		transformDelta.ptr<float>(1)[2] = x.ptr<float>(4)[0];
		transformDelta.ptr<float>(2)[1] = -x.ptr<float>(4)[0];
		transformDelta.ptr<float>(0)[3] = x.ptr<float>(0)[0];
		transformDelta.ptr<float>(1)[3] = x.ptr<float>(1)[0];
		transformDelta.ptr<float>(2)[3] = x.ptr<float>(2)[0];

		final_transform_delta = transformDelta * final_transform_delta;

		cv::Mat cmp = transformDelta - cv::Mat::eye(4, 4, CV_32F);
		float diff = cv::norm(cmp.col(0)) + cv::norm(cmp.col(1)) + cv::norm(cmp.col(2)) + cv::norm(cmp.col(3));
		if (diff < 0.001) break;
	}

	return final_transform_delta;
}