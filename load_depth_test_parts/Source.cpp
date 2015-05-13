#include <Windows.h>
#include <opencv2\opencv.hpp>
#include <cv_skeleton.h>
#include <recons_optimization.h>
#include <ReconsVoxel.h>
#include <cv_pointmat_common.h>

//debug
#include <cv_draw_common.h>

#define SKELETON_CONSTRAINT_WEIGHT 1500
#define NEGATIVE_DEPTH 0 //TODO. figure this out

//KINECT
#define CYLINDER_FITTING_THRESHOLD 0.1
#define CYLINDER_FITTING_RADIUS_MAX 0.3
#define CYLINDER_FITTING_RADIUS_INC 0.05
#define VOXEL_SIZE 0.01
#define DEPTH_MULTIPLIER 1
#define TSDF_MU 0.001

//ASSIMP
//#define CYLINDER_FITTING_THRESHOLD 0.4
//#define CYLINDER_FITTING_RADIUS_MAX 2
//#define CYLINDER_FITTING_RADIUS_INC 0.1
//#define VOXEL_SIZE 0.1
//#define DEPTH_MULTIPLIER -1
//#define TSDF_MU 0.01

#define RELATIVE_SKELETON 1

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

//this version instantly applies transforms and recalculates child transformations immediately after estimation
void estimate_skeleton_and_transform(
	const BodyPartDefinitionVector& bpdv,
	const FrameData& source_framedata,
	const FrameData& target_framedata,
	SkeletonNodeHardMap& source_snhmap,
	const SkeletonNodeHardMap& target_snhmap,
	const std::vector<VoxelMatrix>& source_volumes,
	float voxel_size,
	cv::Mat& debug_img = cv::Mat()){

	bool rel_skel = RELATIVE_SKELETON;

	std::vector<cv::Mat> bodypart_transforms(bpdv.size());

	static int debug_frame = 0;

	std::stringstream ss;

		for (int i = 0; i < bpdv.size(); ++i){

	for(int iter=0;iter<50;++iter){ //straight up lets just do 50 iterations

			ss.str("");
			cv::FileStorage fs;
			ss << "LDTP-debug/debug-frame" << debug_frame << "-bp" << i << ".yml";
			fs.open(ss.str(), cv::FileStorage::WRITE);

			const SkeletonNodeHard * source_node = source_snhmap.find(bpdv[i].mNode1Name)->second;
			const SkeletonNodeHard * target_node = target_snhmap.find(bpdv[i].mNode1Name)->second;

			cv::Mat source_bodypart_transform = get_bodypart_transform(bpdv[i], source_snhmap, source_framedata.mmCameraPose);
			cv::Mat target_bodypart_transform = get_bodypart_transform(bpdv[i], target_snhmap, target_framedata.mmCameraPose);

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

				
				if (source_points.size() > 0){
					//convert to screen coordinates
					source_bodypart_pointmat = source_screen_transform * source_bodypart_pointmat;
				}
			}


			cv::Mat A(6, 6, CV_64F, cv::Scalar(0));
			cv::Mat b(6, 1, CV_64F, cv::Scalar(0));


			//cv::Mat source_parent_bodypart_transform_inv = (source_bodypart_transform * source_node->mTransformation.inv()).inv();
			//cv::Mat target_parent_bodypart_transform_inv = (target_bodypart_transform * target_node->mTransformation.inv()).inv();

			//point to plane
			if (!source_bodypart_pointmat.empty())
			{
				cv::Mat _A, _b;

				if (false){
					point_to_plane_registration(source_bodypart_pointmat, source_framedata.mmDepth, source_framedata.mmCameraMatrix, source_bodypart_transform_inv,
						target_framedata.mmDepth, target_framedata.mmCameraMatrix,
						target_bodypart_transform_inv, voxel_size, _A, _b, true);
				}
				else{
					//try multiplying the inverse camera pose instead of the body part transform
					point_to_plane_registration(source_bodypart_pointmat, source_framedata.mmDepth, source_framedata.mmCameraMatrix, source_framedata.mmCameraPose.inv(),
						target_framedata.mmDepth, target_framedata.mmCameraMatrix,
						target_framedata.mmCameraPose.inv(), voxel_size, _A, _b, true);
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
			//if (!source_bodypart_pointmat.empty())
			//{
			//	cv::Mat _A, _b;
			//
			//	if (false){
			//		point_to_point_registration(source_bodypart_pointmat, source_framedata.mmColor, source_framedata.mmDepth, source_framedata.mmCameraMatrix, source_bodypart_transform_inv,
			//			target_framedata.mmColor, target_framedata.mmDepth, target_framedata.mmCameraMatrix, target_bodypart_transform_inv, _A, _b, true);
			//	}
			//	else{
			//		point_to_point_registration(source_bodypart_pointmat, source_framedata.mmColor, source_framedata.mmDepth, source_framedata.mmCameraMatrix, source_framedata.mmCameraPose.inv(),
			//			target_framedata.mmColor, target_framedata.mmDepth, target_framedata.mmCameraMatrix, target_framedata.mmCameraPose.inv(), _A, _b, true);
			//	}
			//	//point_to_point_registration(source_bodypart_pointmat, source_framedata.mmColor, source_framedata.mmDepth, source_framedata.mmCameraMatrix, source_parent_bodypart_transform_inv,
			//	//	target_framedata.mmColor, target_framedata.mmDepth, target_framedata.mmCameraMatrix, target_parent_bodypart_transform_inv, _A, _b, true);
			//
			//	cv::Mat A64, b64;
			//	_A.convertTo(A64, CV_64F);
			//	_b.convertTo(b64, CV_64F);
			//
			//	cv::add(A, A64, A);
			//	cv::add(b, b64, b);
			//
			//	//debug
			//
			//	cv::Mat x;
			//
			//	cv::solve(A64, b64, x, cv::DECOMP_CHOLESKY);
			//	float energy = cv::norm(A*x - b);
			//	cv::Mat transformDelta = cv::Mat::eye(4, 4, CV_32F);
			//
			//	transformDelta.ptr<float>(0)[1] = x.ptr<double>(3)[0];
			//	transformDelta.ptr<float>(1)[0] = -x.ptr<double>(3)[0];
			//	transformDelta.ptr<float>(0)[2] = -x.ptr<double>(5)[0];
			//	transformDelta.ptr<float>(2)[0] = x.ptr<double>(5)[0];
			//	transformDelta.ptr<float>(1)[2] = x.ptr<double>(4)[0];
			//	transformDelta.ptr<float>(2)[1] = -x.ptr<double>(4)[0];
			//	transformDelta.ptr<float>(0)[3] = x.ptr<double>(0)[0];
			//	transformDelta.ptr<float>(1)[3] = x.ptr<double>(1)[0];
			//	transformDelta.ptr<float>(2)[3] = x.ptr<double>(2)[0];
			//	std::cout << "point to point transform: \n" << transformDelta << std::endl;
			//
			//	fs << "Point_to_point" << "{"
			//		<< "A" << A64
			//		<< "b" << b64
			//		<< "x" << x
			//		<< "transform" << transformDelta
			//		<< "}";
			//}

			//skeleton
			{
				cv::Mat _A, _b;

				if (rel_skel){
					skeleton_constraints_linear(source_node->mTransformation, target_node->mTransformation, 1, 1, _A, _b);
				}
				else{
					skeleton_constraints_linear(source_bodypart_transform, target_bodypart_transform, 1, 1, _A, _b);
				}
				cv::Mat A64, b64;
				_A.convertTo(A64, CV_64F);
				_b.convertTo(b64, CV_64F);

				cv::add(A, target_node->confidence * SKELETON_CONSTRAINT_WEIGHT * A64, A);
				cv::add(b, target_node->confidence * SKELETON_CONSTRAINT_WEIGHT * b64, b);

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
			std::cout << "final transform: \n" << transformDelta << std::endl;


			fs << "Final" << "{"
				<< "A" << A
				<< "b" << b
				<< "x" << x
				<< "transform" << transformDelta
				<< "}";

			fs.release();

			SkeletonNodeHard * node = source_snhmap.find(bpdv[i].mNode1Name)->second;
			SkeletonNodeHard * node_parent = source_snhmap.find(node->mParentName)->second;

			if (rel_skel){
				node->mTransformation = transformDelta * node->mTransformation;
				cv_draw_and_build_skeleton(node, node_parent->mTempTransformation, source_framedata.mmCameraMatrix, source_framedata.mmCameraPose, &source_snhmap);
			}
			else{
				node->mTempTransformation = transformDelta * node->mTempTransformation;
				node->mTransformation = node->mTempTransformation * node_parent->mTempTransformation.inv();
			}

			if (!debug_img.empty()){

				cv::Mat debug_img_volumes = debug_img.clone();

				for (int i = 0; i < bpdv.size(); ++i){
					cv::Vec3b color(bpdv[i].mColor[2] * 255, bpdv[i].mColor[1] * 255, bpdv[i].mColor[0] * 255);
					voxel_draw_volume(debug_img_volumes, color, get_bodypart_transform(bpdv[i], source_snhmap, source_framedata.mmCameraPose), source_framedata.mmCameraMatrix, &source_volumes[i], voxel_size);
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

	}

	++debug_frame;
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

	//std::vector<PointMap> point_maps;
	//std::vector<FrameData> frame_datas;

	//load_frames(filenames, point_maps, frame_datas, true, false);
	//std::vector<SkeletonNodeHardMap> snhmaps;
	//
	//for (int i = 0; i < frame_datas.size(); ++i){
	//	snhmaps.push_back(SkeletonNodeHardMap());
	//	cv_draw_and_build_skeleton(&frame_datas[i].mmRoot, frame_datas[i].mmCameraPose, frame_datas[i].mmCameraMatrix, &snhmaps[i]);
	//
	//
	//	//todo: put some background behind our dude
	//	//for now: just set depth values to some wall number
	//
	//	cv::Mat& depthMat = frame_datas[i].mmDepth;
	//
	//	for (int i = 0; i < depthMat.rows*depthMat.cols; ++i){
	//		if (depthMat.ptr<float>()[i] == 0){
	//			depthMat.ptr<float>()[i] = -4;
	//		}
	//	}
	//}

	float curr_frame_f = 0;

	SkeletonNodeHard prev_root;
	FrameData prev_framedata;
	std::vector<VoxelMatrix> volumes;
	std::vector<cv::Mat> TSDF_array;
	std::vector<cv::Mat> weight_array;
	VoxelSetMap voxelmap;
	std::vector<Cylinder> cylinders;

	bool first_frame = true;

	while (curr_frame_f < filenames.size()){
		unsigned int curr_frame = curr_frame_f;

		SkeletonNodeHard gen_root;

		std::vector<PointMap> pointmap_1frame;
		std::vector<FrameData> framedata_1frame;

		std::vector<std::string> filename_1frame;
		filename_1frame.push_back(filenames[curr_frame]);

		load_frames(filename_1frame, pointmap_1frame, framedata_1frame);
		if (framedata_1frame.empty()) break;

		//for (int i = 0; i < framedata_1frame[0].mmDepth.rows*framedata_1frame[0].mmDepth.cols; ++i){
		//	if (framedata_1frame[0].mmDepth.ptr<float>()[i] == 0){
		//		framedata_1frame[0].mmDepth.ptr<float>()[i] = NEGATIVE_DEPTH? -10 : 10;
		//	}
		//}

		SkeletonNodeHardMap snhmap_currframe;
		cv::Mat test_img = framedata_1frame[0].mmColor.clone();
		cv_draw_and_build_skeleton(&framedata_1frame[0].mmRoot, cv::Mat::eye(4,4,CV_32F), framedata_1frame[0].mmCameraMatrix, framedata_1frame[0].mmCameraPose, &snhmap_currframe, test_img);

		cv::imshow("test_img", test_img);




		if (first_frame){
			first_frame = false;
			gen_root = framedata_1frame[0].mmRoot;

			PointMap pointMap(framedata_1frame[0].width, framedata_1frame[0].height);
			read_depth_image(framedata_1frame[0].mmDepth, framedata_1frame[0].mmCameraMatrix, pointMap);
			cv::Mat pointmat(4, pointMap.mvPointLocations.size(), CV_32F);
			read_points_pointcloud(pointMap, pointmat);
		

			float width = framedata_1frame[0].mmDepth.cols;
			float height = framedata_1frame[0].mmDepth.rows;

			cylinder_fitting(bpdv, snhmap_currframe, pointmat, framedata_1frame[0].mmCameraPose, cylinders, CYLINDER_FITTING_RADIUS_INC, CYLINDER_FITTING_RADIUS_MAX, CYLINDER_FITTING_THRESHOLD, &volume_sizes
				);// , &framedata_1frame[0].mmCameraMatrix, &width, &height);
			init_voxel_set(bpdv, snhmap_currframe, cylinders, framedata_1frame[0].mmCameraPose, volumes, volume_sizes, voxelmap, voxel_size);

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

			cv_draw_and_build_skeleton(&gen_root, cv::Mat::eye(4,4,CV_32F), prev_framedata.mmCameraMatrix, prev_framedata.mmCameraPose, &curr_snhmap);

			//std::vector<cv::Mat> bodypart_transforms = estimate_skeleton(bpdv, frame_datas[curr_frame - 1], frame_datas[curr_frame], curr_snhmap, snhmaps[curr_frame], volumes, voxel_size);
			//for (int i = 0; i < bpdv.size(); ++i){
			//	curr_snhmap.find(bpdv[i].mNode1Name)->second->mTransformation = bodypart_transforms[i] * curr_snhmap.find(bpdv[i].mNode1Name)->second->mTransformation;
			//}

			estimate_skeleton_and_transform(bpdv, prev_framedata, framedata_1frame[0], curr_snhmap, snhmap_currframe, volumes, voxel_size, framedata_1frame[0].mmColor);
		}
		
		filenameSS.str("");
		filenameSS << volume_transform_output_directory << curr_frame + startframe << ".xml.gz";

		save_input_frame(filenameSS.str(), curr_frame_f, framedata_1frame[0].mmCameraPose, framedata_1frame[0].mmCameraMatrix, gen_root, framedata_1frame[0].mmColor, framedata_1frame[0].mmDepth);

		cv::Mat skeleton_image = framedata_1frame[0].mmColor.clone();
		SkeletonNodeHardMap curr_snhmap;

		cv_draw_and_build_skeleton(&gen_root, cv::Mat::eye(4,4,CV_32F), framedata_1frame[0].mmCameraMatrix, framedata_1frame[0].mmCameraPose, &curr_snhmap, skeleton_image);

		//integrate_volume(bpdv, curr_snhmap, cylinders, frame_datas[curr_frame].mmDepth, frame_datas[curr_frame].mmCameraPose, frame_datas[curr_frame].mmCameraMatrix,
		//	volumes, TSDF_array, weight_array, voxel_size);

		std::vector<cv::Mat> bodypart_transforms(bpdv.size());
		for (int i = 0; i < bpdv.size(); ++i){
			bodypart_transforms[i] = get_bodypart_transform(bpdv[i], snhmap_currframe, framedata_1frame[0].mmCameraPose);
		}

		std::vector<Grid3D<char>> voxel_assignments = assign_voxels_to_body_parts(bpdv, bodypart_transforms, cylinders, framedata_1frame[0].mmDepth, framedata_1frame[0].mmCameraPose, framedata_1frame[0].mmCameraMatrix, volumes, voxel_size);

		cv::Mat camera_intrinsic_inv = framedata_1frame[0].mmCameraMatrix.inv();
		cv::Mat camera_extrinsic_inv = framedata_1frame[0].mmCameraPose.inv();

		for (int i = 0; i < bpdv.size(); ++i){
			if (get_skeleton_node(bpdv[i], snhmap_currframe)->confidence > 0.8){
				integrate_volume(bodypart_transforms[i], voxel_assignments[i], framedata_1frame[0].mmDepth, framedata_1frame[0].mmCameraPose, camera_extrinsic_inv, framedata_1frame[0].mmCameraMatrix, camera_intrinsic_inv, volumes[i], TSDF_array[i], weight_array[i], voxel_size, TSDF_MU, DEPTH_MULTIPLIER);
			}
		}

		for (int i = 0; i < bpdv.size(); ++i){
			cv::Vec3b color(bpdv[i].mColor[2] * 255, bpdv[i].mColor[1] * 255, bpdv[i].mColor[0] * 255);
			voxel_draw_volume(skeleton_image, color, get_bodypart_transform(bpdv[i], curr_snhmap, framedata_1frame[0].mmCameraPose), framedata_1frame[0].mmCameraMatrix, &volumes[i], voxel_size);
		}

		cv::imshow("skeleton", skeleton_image);

		cv::waitKey(20);
		prev_root = gen_root;
		prev_framedata = framedata_1frame[0];
		//++curr_frame;
		curr_frame_f += 1;
	}


	std::stringstream voxel_recons_SS;
	voxel_recons_SS << video_directory << "/voxels.xml.gz";
	save_voxels(voxel_recons_SS.str(), cylinders, volumes, TSDF_array, weight_array, voxel_size);
}