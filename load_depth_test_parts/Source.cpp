#include <Windows.h>
#include <opencv2\opencv.hpp>
#include <cv_skeleton.h>
#include <recons_optimization.h>
#include <ReconsVoxel.h>
#include <cv_pointmat_common.h>
#include <recons_registration.h>

//debug
#include <cv_draw_common.h>

#define NEGATIVE_DEPTH 1 //TODO. figure this out

//KINECT
#define CYLINDER_FITTING_THRESHOLD 0.1
#define CYLINDER_FITTING_RADIUS_MAX 0.3
#define CYLINDER_FITTING_RADIUS_INC 0.05
#define VOXEL_SIZE 0.01
#define DEPTH_MULTIPLIER -1
#define TSDF_MU 0.001

//ASSIMP
//#define CYLINDER_FITTING_THRESHOLD 0.4
//#define CYLINDER_FITTING_RADIUS_MAX 2
//#define CYLINDER_FITTING_RADIUS_INC 0.1
//#define VOXEL_SIZE 0.1
//#define DEPTH_MULTIPLIER -1
//#define TSDF_MU 0.01


#if 0
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
		
			point_to_point_registration(source_bodypart_pointmat, source_framedata.mmColor, source_framedata.mmDepth, source_framedata.mmCameraMatrix, get_bodypart_transform(bpdv[i], source_snhmap).inv(),
				target_framedata.mmColor, target_framedata.mmDepth, target_framedata.mmCameraMatrix, get_bodypart_transform(bpdv[i], target_snhmap).inv(), _A, _b);
		
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
#endif


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

	float voxel_size = VOXEL_SIZE;


	if (argc <= 1){
		std::cout << "Please enter directory\n";
		return 0;
	}

	std::string video_directory(argv[1]);

	std::string volume_transform_output_directory = video_directory + "/volume_transform/";
	CreateDirectory(volume_transform_output_directory.c_str(), NULL);

	std::stringstream filenameSS;
	int startframe = 0;
	int numframes = 200;
	cv::FileStorage fs;

	if (argc >= 3){
		numframes = atoi(argv[2]);
	}


	filenameSS << video_directory << "/bodypartdefinitions.xml.gz";

	fs.open(filenameSS.str(), cv::FileStorage::READ);

	if (!fs.isOpened()){
		std::cout << "ERROR: no body part definitions found!\n";
		return 0;
	}

	BodyPartDefinitionVector bpdv;
	for (auto it = fs["bodypartdefinitions"].begin();
		it != fs["bodypartdefinitions"].end();
		++it){
		BodyPartDefinition bpd;
		read(*it, bpd);
		bpdv.push_back(bpd);
	}
	fs.release();

	filenameSS.str("");
	filenameSS << video_directory << "/volume_transform/bodypartdefinitions.xml.gz";
	fs.open(filenameSS.str(), cv::FileStorage::WRITE);
	fs << "bodypartdefinitions" << "[";
	for (int i = 0; i < bpdv.size();++i)
	{
		fs << (bpdv[i]);
	}
	fs << "]";
	fs.release();

	filenameSS.str("");
	filenameSS << video_directory << "/customvolumes.xml.gz";
	fs.open(filenameSS.str(), cv::FileStorage::READ);

	if (!fs.isOpened()){
		std::cout << "ERROR: no custom volume sizes found!\n";
		return 0;
	}

	std::vector<VolumeDimensions> volume_sizes;
	for (auto it = fs["customvolumes"].begin();
		it != fs["customvolumes"].end();
		++it){
		VolumeDimensions vd(0, 0, 0);
		(*it)["width"] >> vd.width;
		(*it)["height"] >> vd.height;
		(*it)["depth"] >> vd.depth;
		volume_sizes.push_back(vd);
	}


	std::vector<std::string> filenames;

	for (int frame = startframe; frame < startframe + numframes; ++frame){
		filenameSS.str("");
		filenameSS << video_directory << "/" << frame << ".xml.gz";

		filenames.push_back(filenameSS.str());

	}

	float curr_frame_f = 0;

	SkeletonNodeAbsoluteVector snav_prev;

	cv::Mat camera_matrix_prev, camera_pose_prev;
	cv::Mat color_prev, depth_prev;

	std::vector<VoxelMatrix> volumes;
	std::vector<cv::Mat> TSDF_array;
	std::vector<cv::Mat> weight_array;
	VoxelSetMap voxelmap;
	std::vector<Cylinder> cylinders;

	bool first_frame = true;
	int facing_prev = FACING_FRONT;


	//register to background
	std::vector<cv::Mat> depth_mats, rgb_mats, camera_mats, camera_poses;

	while (curr_frame_f < filenames.size()){
		unsigned int curr_frame = curr_frame_f;

		SkeletonNodeHard relroot_load;
		SkeletonNodeAbsoluteVector snav_load, snav_gen;

		double time;
		cv::Mat camera_matrix_load, camera_pose_load;
		cv::Mat color_load, fullcolor_load, depth_load;
		int facing;

		bool load_success = load_input_frame(filenames[curr_frame], time, camera_pose_load, camera_matrix_load, relroot_load, color_load, fullcolor_load, depth_load, facing);
		if (!load_success) break;

		float width = color_load.cols, height = color_load.rows;

		absolutize_snh(relroot_load, snav_load);

		cv::Mat test_img = color_load.clone();
		cv::imshow("test_img", test_img);

		depth_mats.push_back(depth_load.clone());
		rgb_mats.push_back(fullcolor_load.clone());
		camera_poses.push_back(cv::Mat::eye(4, 4, CV_32F));
		camera_mats.push_back(camera_matrix_load);

		cv::Mat bg_transform_delta = cv::Mat::eye(4, 4, CV_32F);
		if (first_frame){
			first_frame = false;

			depth_mats.push_back(depth_load.clone());
			rgb_mats.push_back(fullcolor_load.clone());
			camera_poses.push_back(cv::Mat::eye(4, 4, CV_32F));
			camera_mats.push_back(camera_matrix_load);

			snav_gen = snav_load;

			PointMap pointMap(width, height);
			read_depth_image(depth_load, camera_matrix_load, pointMap);
			cv::Mat pointmat(4, pointMap.mvPointLocations.size(), CV_32F);
			read_points_pointcloud(pointMap, pointmat);

			cylinder_fitting(bpdv, snav_load, pointmat, camera_pose_load, cylinders, CYLINDER_FITTING_RADIUS_INC, CYLINDER_FITTING_RADIUS_MAX, CYLINDER_FITTING_THRESHOLD, &volume_sizes
				);// , &camera_matrix_load, &width, &height);
			init_voxel_set(bpdv, snav_load, cylinders, camera_pose_load, volumes, volume_sizes, voxelmap, voxel_size);
			

			TSDF_array.reserve(volumes.size());
			weight_array.reserve(volumes.size());
			for (int i = 0; i < volumes.size(); ++i){
				int size = volumes[i].width * volumes[i].height * volumes[i].depth;
				TSDF_array.push_back(cv::Mat(1, size, CV_32F, cv::Scalar(0)));
				weight_array.push_back(cv::Mat(1, size, CV_32F, cv::Scalar(0)));
			}
		}
		else if (facing == FACING_SIDE || facing_prev == FACING_SIDE){
			snav_gen = snav_load;
			//snav_gen = snav_prev;
			//
			//cv::Mat depth_as_color_1;
			//cv::normalize(depth_load, depth_as_color_1, 0, 0xff, CV_MINMAX, CV_8U);
			//cv::Mat depth_as_color_3;
			//cv::cvtColor(depth_as_color_1, depth_as_color_3, CV_GRAY2BGR);
			//
			//estimate_skeleton_and_transform(bpdv, camera_matrix_prev, camera_pose_prev, color_prev, depth_prev, camera_matrix_load, camera_pose_load, color_load, depth_load, snav_gen, snav_load, volumes, voxel_size, depth_as_color_3);
			//
			//*get_bodypart_skeleton_node("UPPER ARM RIGHT", bpdv, snav_gen) = *get_bodypart_skeleton_node("UPPER ARM RIGHT", bpdv, snav_load);
			//*get_bodypart_skeleton_node("LOWER ARM RIGHT", bpdv, snav_gen) = *get_bodypart_skeleton_node("LOWER ARM RIGHT", bpdv, snav_load);
			//*get_bodypart_skeleton_node("UPPER LEG RIGHT", bpdv, snav_gen) = *get_bodypart_skeleton_node("UPPER LEG RIGHT", bpdv, snav_load);
			//*get_bodypart_skeleton_node("LOWER LEG RIGHT", bpdv, snav_gen) = *get_bodypart_skeleton_node("LOWER LEG RIGHT", bpdv, snav_load);
			//*get_bodypart_skeleton_node("UPPER ARM LEFT", bpdv, snav_gen) = *get_bodypart_skeleton_node("UPPER ARM LEFT", bpdv, snav_load);
			//*get_bodypart_skeleton_node("LOWER ARM LEFT", bpdv, snav_gen) = *get_bodypart_skeleton_node("LOWER ARM LEFT", bpdv, snav_load);
			//*get_bodypart_skeleton_node("UPPER LEG LEFT", bpdv, snav_gen) = *get_bodypart_skeleton_node("UPPER LEG LEFT", bpdv, snav_load);
			//*get_bodypart_skeleton_node("LOWER LEG LEFT", bpdv, snav_gen) = *get_bodypart_skeleton_node("LOWER LEG LEFT", bpdv, snav_load);
		}
		else if (facing == FACING_SIDE_BACK || facing_prev == FACING_SIDE_BACK){
			snav_gen = snav_load;
			//snav_gen = snav_prev;
			//
			//cv::Mat depth_as_color_1;
			//cv::normalize(depth_load, depth_as_color_1, 0, 0xff, CV_MINMAX, CV_8U);
			//cv::Mat depth_as_color_3;
			//cv::cvtColor(depth_as_color_1, depth_as_color_3, CV_GRAY2BGR);
			//
			//estimate_skeleton_and_transform(bpdv, camera_matrix_prev, camera_pose_prev, color_prev, depth_prev, camera_matrix_load, camera_pose_load, color_load, depth_load, snav_gen, snav_load, volumes, voxel_size, depth_as_color_3);
			//
			//
			//*get_bodypart_skeleton_node("UPPER ARM RIGHT", bpdv, snav_gen) = *get_bodypart_skeleton_node("UPPER ARM RIGHT", bpdv, snav_load);
			//*get_bodypart_skeleton_node("LOWER ARM RIGHT", bpdv, snav_gen) = *get_bodypart_skeleton_node("LOWER ARM RIGHT", bpdv, snav_load);
			//*get_bodypart_skeleton_node("UPPER LEG RIGHT", bpdv, snav_gen) = *get_bodypart_skeleton_node("UPPER LEG RIGHT", bpdv, snav_load);
			//*get_bodypart_skeleton_node("LOWER LEG RIGHT", bpdv, snav_gen) = *get_bodypart_skeleton_node("LOWER LEG RIGHT", bpdv, snav_load);
			//*get_bodypart_skeleton_node("UPPER ARM LEFT", bpdv, snav_gen) = *get_bodypart_skeleton_node("UPPER ARM LEFT", bpdv, snav_load);
			//*get_bodypart_skeleton_node("LOWER ARM LEFT", bpdv, snav_gen) = *get_bodypart_skeleton_node("LOWER ARM LEFT", bpdv, snav_load);
			//*get_bodypart_skeleton_node("UPPER LEG LEFT", bpdv, snav_gen) = *get_bodypart_skeleton_node("UPPER LEG LEFT", bpdv, snav_load);
			//*get_bodypart_skeleton_node("LOWER LEG LEFT", bpdv, snav_gen) = *get_bodypart_skeleton_node("LOWER LEG LEFT", bpdv, snav_load);
		}
		else{

			snav_gen = snav_prev;
			//bg_transform_delta = estimate_background_transform_multi(depth_load, fullcolor_load, camera_matrix_load, depth_mats, rgb_mats, camera_mats, camera_poses);
			camera_pose_load = bg_transform_delta.inv();

			cv::Mat depth_as_color_1;
			cv::normalize(depth_load, depth_as_color_1, 0, 0xff, CV_MINMAX, CV_8U);
			cv::Mat depth_as_color_3;
			cv::cvtColor(depth_as_color_1, depth_as_color_3, CV_GRAY2BGR);

			estimate_skeleton_and_transform(bpdv, camera_matrix_prev, camera_pose_prev, color_prev, depth_prev, camera_matrix_load, camera_pose_load, color_load, depth_load, snav_gen, snav_load, volumes, voxel_size, depth_as_color_3);
		}

		depth_mats.pop_back();
		rgb_mats.pop_back();
		camera_mats.pop_back();
		camera_poses.pop_back();
		
		filenameSS.str("");
		filenameSS << volume_transform_output_directory << curr_frame + startframe << ".xml.gz";

		SkeletonNodeHard root_rel;
		relativize_snh(snav_gen, root_rel);

		save_input_frame(filenameSS.str(), curr_frame_f, camera_pose_load, camera_matrix_load, root_rel, color_load, fullcolor_load, depth_load, facing);

		cv::Mat skeleton_image = color_load.clone();
		SkeletonNodeHardMap curr_snhmap;

		cv_draw_and_build_skeleton(&root_rel, cv::Mat::eye(4,4,CV_32F), camera_matrix_load, camera_pose_load, &curr_snhmap, skeleton_image);

		//integrate_volume(bpdv, curr_snhmap, cylinders, frame_datas[curr_frame].mmDepth, frame_datas[curr_frame].mmCameraPose, frame_datas[curr_frame].mmCameraMatrix,
		//	volumes, TSDF_array, weight_array, voxel_size);

		std::vector<cv::Mat> bodypart_transforms(bpdv.size());
		for (int i = 0; i < bpdv.size(); ++i){
			bodypart_transforms[i] = get_bodypart_transform(bpdv[i], curr_snhmap, camera_pose_load);
		}

		std::vector<Grid3D<char>> voxel_assignments = assign_voxels_to_body_parts(bpdv, bodypart_transforms, cylinders, depth_load, camera_pose_load, camera_matrix_load, volumes, voxel_size);

		cv::Mat camera_intrinsic_inv = camera_matrix_load.inv();
		cv::Mat camera_extrinsic_inv = camera_pose_load.inv();

		for (int i = 0; i < bpdv.size(); ++i){

			//if (bpdv[i].mBodyPartName.length() > 5){
			//
			//	if ((bpdv[i].mBodyPartName.substr(bpdv[i].mBodyPartName.length() - 4) == "LEFT" || bpdv[i].mBodyPartName.substr(bpdv[i].mBodyPartName.length() - 5) == "RIGHT") && (facing == FACING_SIDE || facing == FACING_SIDE_BACK)){
			//		continue;
			//	}
			//}

			if (facing == FACING_SIDE || facing == FACING_SIDE_BACK ||
				facing_prev == FACING_SIDE || facing_prev == FACING_SIDE_BACK){
				continue;
			}

			if (get_skeleton_node(bpdv[i], snav_load)->confidence > 0.8){
				integrate_volume(bodypart_transforms[i], voxel_assignments[i], depth_load, camera_pose_load, camera_extrinsic_inv, camera_matrix_load, camera_intrinsic_inv, volumes[i], TSDF_array[i], weight_array[i], voxel_size, TSDF_MU, DEPTH_MULTIPLIER);
			}
		}

		for (int i = 0; i < bpdv.size(); ++i){
			cv::Vec3b color(bpdv[i].mColor[2] * 255, bpdv[i].mColor[1] * 255, bpdv[i].mColor[0] * 255);
			voxel_draw_volume(skeleton_image, color, get_bodypart_transform(bpdv[i], curr_snhmap, camera_pose_load), camera_matrix_load, &volumes[i], voxel_size);
		}

		cv::imshow("skeleton", skeleton_image);

		cv::waitKey(20);
		snav_prev = snav_gen;

		camera_pose_prev = camera_pose_load;
		camera_matrix_prev = camera_matrix_load;
		color_prev = color_load;
		depth_prev = depth_load;
		facing_prev = facing;
		//++curr_frame;
		curr_frame_f += 1;

		//debug: saves voxels at each frame

		std::stringstream voxel_recons_SS;
		voxel_recons_SS << video_directory << "/voxels-frame" << curr_frame << ".xml.gz";
		save_voxels(voxel_recons_SS.str(), cylinders, volumes, TSDF_array, weight_array, voxel_size);
	}


	std::stringstream voxel_recons_SS;
	voxel_recons_SS << video_directory << "/voxels.xml.gz";
	save_voxels(voxel_recons_SS.str(), cylinders, volumes, TSDF_array, weight_array, voxel_size);
}