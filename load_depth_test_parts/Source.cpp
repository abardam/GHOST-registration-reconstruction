#include <opencv2\opencv.hpp>
#include <recons_optimization.h>
#include <AssimpOpenGL.h>
#include <ReconsVoxel.h>
#include <cv_pointmat_common.h>

#define SKELETON_CONSTRAINT_WEIGHT 3000

std::vector<cv::Mat> estimate_skeleton(
	const BodyPartDefinitionVector& bpdv, 
	const FrameData& source_framedata, 
	const FrameData& target_framedata, 
	const SkeletonNodeHardMap& source_snhmap, 
	const SkeletonNodeHardMap& target_snhmap,
	const std::vector<VoxelMatrix>& source_volumes,
	float voxel_size){

	std::vector<cv::Mat> bodypart_transforms(bpdv.size());


	for (int i = 0; i < bpdv.size(); ++i){

		cv::Mat source_screen_transform = get_bodypart_transform(bpdv[i], source_snhmap) * get_voxel_transform(source_volumes[i].width, source_volumes[i].height, source_volumes[i].depth, voxel_size);

		cv::Mat source_bodypart_pointmat;
		{
			std::vector<cv::Vec4f> source_points;
			for (int j = 0; j < source_volumes[i].voxel_data.cols; ++j){
				if (source_volumes[i].voxel_data.ptr<cv::Vec4b>()[j](3) == 0xff){
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

			//convert to screen coordinates
			source_bodypart_pointmat = source_screen_transform * source_bodypart_pointmat;
		}


		cv::Mat A(6, 6, CV_32F, cv::Scalar(0));
		cv::Mat b(6, 1, CV_32F, cv::Scalar(0));

		//point to plane
		{
			cv::Mat _A, _b;
		
			point_to_plane_registration(source_bodypart_pointmat, source_framedata.mmDepth, source_framedata.mmCameraMatrix, get_bodypart_transform(bpdv[i], source_snhmap).inv(),
				target_framedata.mmDepth, target_framedata.mmCameraMatrix, get_bodypart_transform(bpdv[i], target_snhmap).inv(), voxel_size, _A, _b, true);
		
			cv::add(A, _A, A);
			cv::add(b, _b, b);
		}

		//point to point
		{
			cv::Mat _A, _b;
		
			point_to_point_registration(source_bodypart_pointmat, source_framedata.mmColor, source_framedata.mmCameraMatrix, get_bodypart_transform(bpdv[i], source_snhmap).inv(),
				target_framedata.mmColor, target_framedata.mmDepth, target_framedata.mmCameraMatrix, _A, _b);
		
			cv::add(A, _A, A);
			cv::add(b, _b, b);
		}

		//skeleton
		{
			cv::Mat _A, _b;
			
			const SkeletonNodeHard * source_node = source_snhmap.find(bpdv[i].mNode1Name)->second;
			const SkeletonNodeHard * target_node = target_snhmap.find(bpdv[i].mNode1Name)->second;

			skeleton_constraints_linear(source_node->mTransformation, target_node->mTransformation, 1, 1, _A, _b);

			cv::add(A, SKELETON_CONSTRAINT_WEIGHT * _A, A);
			cv::add(b, SKELETON_CONSTRAINT_WEIGHT * _b, b);
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

		bodypart_transforms[i] = transformDelta;
	}

	return bodypart_transforms;
}

//this version instantly applies transforms and recalculates child transformations immediately after estimation
void estimate_skeleton_and_transform(
	const BodyPartDefinitionVector& bpdv,
	const FrameData& source_framedata,
	const FrameData& target_framedata,
	SkeletonNodeHardMap& source_snhmap,
	const SkeletonNodeHardMap& target_snhmap,
	const std::vector<VoxelMatrix>& source_volumes,
	float voxel_size){

	std::vector<cv::Mat> bodypart_transforms(bpdv.size());


	for (int i = 0; i < bpdv.size(); ++i){

		cv::Mat source_bodypart_transform = get_bodypart_transform(bpdv[i], source_snhmap);
		cv::Mat target_bodypart_transform = get_bodypart_transform(bpdv[i], target_snhmap);

		cv::Mat source_bodypart_transform_inv = source_bodypart_transform.inv();
		cv::Mat target_bodypart_transform_inv = target_bodypart_transform.inv();

		cv::Mat source_screen_transform = source_bodypart_transform * get_voxel_transform(source_volumes[i].width, source_volumes[i].height, source_volumes[i].depth, voxel_size);

		cv::Mat source_bodypart_pointmat;
		{
			std::vector<cv::Vec4f> source_points;
			for (int j = 0; j < source_volumes[i].voxel_data.cols; ++j){
				if (source_volumes[i].voxel_data.ptr<cv::Vec4b>()[j](3) == 0xff){
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

			//convert to screen coordinates
			source_bodypart_pointmat = source_screen_transform * source_bodypart_pointmat;
		}


		cv::Mat A(6, 6, CV_32F, cv::Scalar(0));
		cv::Mat b(6, 1, CV_32F, cv::Scalar(0));

		//point to plane
		{
			cv::Mat _A, _b;

			point_to_plane_registration(source_bodypart_pointmat, source_framedata.mmDepth, source_framedata.mmCameraMatrix, source_bodypart_transform_inv,
				target_framedata.mmDepth, target_framedata.mmCameraMatrix, 
				target_bodypart_transform_inv, voxel_size, _A, _b, true);

			cv::add(A, _A, A);
			cv::add(b, _b, b);
		}

		//point to point
		{
			cv::Mat _A, _b;

			point_to_point_registration(source_bodypart_pointmat, source_framedata.mmColor, source_framedata.mmCameraMatrix, source_bodypart_transform_inv,
				target_framedata.mmColor, target_framedata.mmDepth, target_framedata.mmCameraMatrix, _A, _b, true);

			cv::add(A, _A, A);
			cv::add(b, _b, b);
		}

		//skeleton
		{
			cv::Mat _A, _b;

			const SkeletonNodeHard * source_node = source_snhmap.find(bpdv[i].mNode1Name)->second;
			const SkeletonNodeHard * target_node = target_snhmap.find(bpdv[i].mNode1Name)->second;

			skeleton_constraints_linear(source_node->mTransformation, target_node->mTransformation, 1, 1, _A, _b);

			cv::add(A, SKELETON_CONSTRAINT_WEIGHT * _A, A);
			cv::add(b, SKELETON_CONSTRAINT_WEIGHT * _b, b);
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

		//SkeletonNodeHard * node = source_snhmap.find(bpdv[i].mNode1Name)->second;
		//node->mTransformation = transformDelta * node->mTransformation;
		//node->mTempTransformation = transformDelta * node->mTempTransformation;
		//for (int i = 0; i < node->mChildren.size(); ++i){
		//	cv_draw_and_build_skeleton(&node->mChildren[i], node->mTempTransformation, source_framedata.mmCameraMatrix, &source_snhmap);
		//}
	}
}

SkeletonNodeHard estimate_skeleton(const SkeletonNodeHard * const ref, const SkeletonNodeHard * const prev){
	SkeletonNodeHard snh;
	cv::Mat generatedTransformation = skeleton_constraints_optimize(prev->mTransformation, ref->mTransformation, 1, 1);


	for (int i = 0; i < prev->mChildren.size(); ++i){
		snh.mChildren.push_back(estimate_skeleton(&ref->mChildren[i], &prev->mChildren[i]));
	}

	snh.mTransformation = generatedTransformation * prev->mTransformation;

	//extract scale and correct it?
	cv::Vec3f scale(
		1 / cv::norm(snh.mTransformation(cv::Range(0, 1), cv::Range(0, 3))),
		1 / cv::norm(snh.mTransformation(cv::Range(1, 2), cv::Range(0, 3))),
		1 / cv::norm(snh.mTransformation(cv::Range(2, 3), cv::Range(0, 3)))
		);

	cv::Mat rot = cv::Mat::diag(cv::Mat(scale)) * snh.mTransformation(cv::Range(0, 3), cv::Range(0, 3));
	rot.copyTo(snh.mTransformation(cv::Range(0, 3), cv::Range(0, 3)));

	snh.mName = prev->mName;
	snh.mParentName = prev->mParentName;

	return snh;
}

int main(int argc, char** argv){

	float voxel_size = 0.1;

	//experimental
	std::vector<std::pair<float, float>> volume_sizes;
	volume_sizes.push_back(std::pair<float, float>(3, 3));
	volume_sizes.push_back(std::pair<float, float>(6, 6));
	volume_sizes.push_back(std::pair<float, float>(5, 5));
	volume_sizes.push_back(std::pair<float, float>(2.5, 2.5));
	volume_sizes.push_back(std::pair<float, float>(2.5, 2.5));
	volume_sizes.push_back(std::pair<float, float>(2, 2));
	volume_sizes.push_back(std::pair<float, float>(2, 2));
	volume_sizes.push_back(std::pair<float, float>(4, 2));
	volume_sizes.push_back(std::pair<float, float>(4, 2));
	volume_sizes.push_back(std::pair<float, float>(3, 3));
	volume_sizes.push_back(std::pair<float, float>(3, 3));
	volume_sizes.push_back(std::pair<float, float>(2, 2));
	volume_sizes.push_back(std::pair<float, float>(2, 2));
	volume_sizes.push_back(std::pair<float, float>(2, 2));
	volume_sizes.push_back(std::pair<float, float>(2, 2));

	if (argc <= 1){
		std::cout << "Please enter directory\n";
		return 0;
	}

	bool point_to_point = false;
	bool point_to_plane = true;
	bool skeleton = false;

	std::string video_directory(argv[1]);
	std::stringstream filenameSS;
	int startframe = 0;
	int numframes = 90;
	cv::FileStorage fs;

	filenameSS << video_directory << "/bodypartdefinitions.xml.gz";

	fs.open(filenameSS.str(), cv::FileStorage::READ);
	BodyPartDefinitionVector bpdv;
	for (auto it = fs["bodypartdefinitions"].begin();
		it != fs["bodypartdefinitions"].end();
		++it){
		BodyPartDefinition bpd;
		read(*it, bpd);
		bpdv.push_back(bpd);
	}
	fs.release();
	std::vector<std::string> filenames;

	for (int frame = startframe; frame < startframe + numframes; ++frame){
		filenameSS.str("");
		filenameSS << video_directory << "/" << frame << ".xml.gz";

		filenames.push_back(filenameSS.str());

	}

	std::vector<PointMap> point_maps;
	std::vector<FrameData> frame_datas;

	load_frames(filenames, point_maps, frame_datas);
	std::vector<SkeletonNodeHardMap> snhmaps;

	for (int i = 0; i < frame_datas.size(); ++i){
		snhmaps.push_back(SkeletonNodeHardMap());
		cv_draw_and_build_skeleton(&frame_datas[i].mmRoot, frame_datas[i].mmCameraPose, frame_datas[i].mmCameraMatrix, &snhmaps[i]);


		//todo: put some background behind our dude
		//for now: just set depth values to some wall number

		cv::Mat& depthMat = frame_datas[i].mmDepth;

		for (int i = 0; i < depthMat.rows*depthMat.cols; ++i){
			if (depthMat.ptr<float>()[i] == 0){
				depthMat.ptr<float>()[i] = -4;
			}
		}
	}

	unsigned int curr_frame = 0;

	SkeletonNodeHard prev_root;
	std::vector<VoxelMatrix> volumes;
	std::vector<cv::Mat> TSDF_array;
	std::vector<cv::Mat> weight_array;
	VoxelSetMap voxelmap;
	std::vector<Cylinder> cylinders;

	while (curr_frame < frame_datas.size()){

		SkeletonNodeHard gen_root;

		if (curr_frame == 0){
			gen_root = frame_datas[curr_frame].mmRoot;

			PointMap pointMap(frame_datas[curr_frame].width, frame_datas[curr_frame].height);
			read_depth_image(frame_datas[curr_frame].mmDepth, frame_datas[curr_frame].mmCameraMatrix, pointMap);
			cv::Mat pointmat(4, pointMap.mvPointLocations.size(), CV_32F);
			read_points_pointcloud(pointMap, pointmat);

			cylinder_fitting(bpdv, snhmaps[curr_frame], pointmat, frame_datas[curr_frame].mmCameraPose, cylinders);
			init_voxel_set(bpdv, snhmaps[curr_frame], cylinders, frame_datas[curr_frame].mmCameraPose, volumes, volume_sizes, voxelmap, voxel_size);

			TSDF_array.reserve(volumes.size());
			weight_array.reserve(volumes.size());
			for (int i = 0; i < volumes.size(); ++i){
				int size = volumes[i].width * volumes[i].height * volumes[i].depth;
				TSDF_array.push_back(cv::Mat(1, size, CV_32F, cv::Scalar(0)));
				weight_array.push_back(cv::Mat(1, size, CV_32F, cv::Scalar(0)));
			}
		}
		else{


			gen_root = prev_root;

			SkeletonNodeHardMap curr_snhmap;

			cv_draw_and_build_skeleton(&gen_root, frame_datas[curr_frame - 1].mmCameraPose, frame_datas[curr_frame - 1].mmCameraMatrix, &curr_snhmap);

			//std::vector<cv::Mat> bodypart_transforms = estimate_skeleton(bpdv, frame_datas[curr_frame - 1], frame_datas[curr_frame], curr_snhmap, snhmaps[curr_frame], volumes, voxel_size);
			//for (int i = 0; i < bpdv.size(); ++i){
			//	curr_snhmap.find(bpdv[i].mNode1Name)->second->mTransformation = bodypart_transforms[i] * curr_snhmap.find(bpdv[i].mNode1Name)->second->mTransformation;
			//}

			estimate_skeleton_and_transform(bpdv, frame_datas[curr_frame - 1], frame_datas[curr_frame], curr_snhmap, snhmaps[curr_frame], volumes, voxel_size);
		}

		cv::Mat skeleton_image = frame_datas[curr_frame].mmColor.clone();
		SkeletonNodeHardMap curr_snhmap;

		cv_draw_and_build_skeleton(&gen_root, frame_datas[curr_frame].mmCameraPose, frame_datas[curr_frame].mmCameraMatrix, &curr_snhmap, skeleton_image);

		//integrate_volume(bpdv, curr_snhmap, cylinders, frame_datas[curr_frame].mmDepth, frame_datas[curr_frame].mmCameraPose, frame_datas[curr_frame].mmCameraMatrix,
		//	volumes, TSDF_array, weight_array, voxel_size);

		std::vector<cv::Mat> bodypart_transforms(bpdv.size());
		for (int i = 0; i < bpdv.size(); ++i){
			bodypart_transforms[i] = get_bodypart_transform(bpdv[i], snhmaps[curr_frame]);
		}

		std::vector<Grid3D<char>> voxel_assignments = assign_voxels_to_body_parts(bpdv, bodypart_transforms, cylinders, frame_datas[curr_frame].mmDepth, frame_datas[curr_frame].mmCameraPose, frame_datas[curr_frame].mmCameraMatrix, volumes, voxel_size);

		cv::Mat camera_intrinsic_inv = frame_datas[curr_frame].mmCameraMatrix.inv();
		cv::Mat camera_extrinsic_inv = frame_datas[curr_frame].mmCameraPose.inv();

		for (int i = 0; i < bpdv.size(); ++i){
			integrate_volume(bodypart_transforms[i], voxel_assignments[i], frame_datas[curr_frame].mmDepth, frame_datas[curr_frame].mmCameraPose, camera_extrinsic_inv, frame_datas[curr_frame].mmCameraMatrix, camera_intrinsic_inv, volumes[i], TSDF_array[i], weight_array[i], voxel_size, 0.01f);
		}

		for (int i = 0; i < bpdv.size(); ++i){
			cv::Vec3b color(bpdv[i].mColor[2] * 255, bpdv[i].mColor[1] * 255, bpdv[i].mColor[0] * 255);
			voxel_draw_volume(skeleton_image, color, get_bodypart_transform(bpdv[i], curr_snhmap), frame_datas[curr_frame].mmCameraMatrix, &volumes[i], voxel_size);
		}

		cv::imshow("skeleton", skeleton_image);

		cv::waitKey(40);
		prev_root = gen_root;
		++curr_frame;
	}


	std::stringstream voxel_recons_SS;
	voxel_recons_SS << video_directory << "/voxels.xml.gz";
	save_voxels(voxel_recons_SS.str(), cylinders, volumes, TSDF_array, weight_array, voxel_size);
}