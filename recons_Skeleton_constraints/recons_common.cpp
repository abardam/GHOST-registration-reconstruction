#include "recons_common.h"


PointMap::PointMap(int w, int h) :
width(w), height(h), mvPoints(w*h, cv::Vec3f(0, 0, 0)){
}

void PointMap::addPoint(int x, int y, const cv::Vec3f& pt){
	if (mvPoints[y*width + x] == cv::Vec3f(0, 0, 0)){
		mvPointLocations.push_back(std::pair<int, int>(x, y));
		mvPoints[y*width + x] = pt;
	}
	else{
		float existing_z = mvPoints[y*width + x](2);
		float new_z = pt(2);
		if (new_z > existing_z) mvPoints[y*width + x] = pt;
	}
}

FrameData::FrameData(int numPts, const cv::Mat& depthMat, const cv::Mat& colorMat, const cv::Mat& cameraMatrix, const SkeletonNodeHard& root) :
mmPoints(4, numPts, CV_32F),
mmNormals(4, numPts, CV_32F),
mmDepth(depthMat.clone()),
mmColor(colorMat.clone()),
mmCameraMatrix(cameraMatrix.clone()),
mmRoot(root),
width(depthMat.cols),
height(depthMat.rows){}

FrameData::FrameData(const FrameData& f) :
mmPoints(f.mmPoints.clone()),
mmNormals(f.mmNormals.clone()),
mmDepth(f.mmDepth.clone()),
mmColor(f.mmColor.clone()),
mmCameraMatrix(f.mmCameraMatrix.clone()),
mmRoot(f.mmRoot),
mmCameraPose(f.mmCameraPose.clone()),
width(mmDepth.cols),
height(mmDepth.rows)
{}


cv::Mat generate_camera_intrinsic(int win_width, int win_height, float fovy){
	cv::Mat camera_intrinsic = cv::Mat::eye(4, 4, CV_32F);
	camera_intrinsic.ptr<float>(0)[0] = -win_width / (2 * tan(AI_DEG_TO_RAD((fovy * (win_width / win_height) / 2.)))); //for some strange reason this is inaccurate for non-square aspect ratios
	camera_intrinsic.ptr<float>(1)[1] = win_height / (2 * tan(AI_DEG_TO_RAD(fovy / 2.)));
	camera_intrinsic.ptr<float>(0)[2] = win_width / 2 + 0.5;
	camera_intrinsic.ptr<float>(1)[2] = win_height / 2 + 0.5;
	return camera_intrinsic;
}

void read_depth_image(const cv::Mat& depthMat, const cv::Mat& camera_intrinsic, PointMap& pointMap){
	cv::Mat camera_intrinsic_inv = camera_intrinsic.inv();
	for (int y = 0; y < depthMat.rows; ++y){
		for (int x = 0; x < depthMat.cols; ++x){
			cv::Vec4f projectedPt(x, y, 1, 1);

			float depth = depthMat.ptr<float>(y)[x];

			if (abs(depth) > FLT_EPSILON){

				cv::Mat reprojectedPt_m = depth * camera_intrinsic_inv * cv::Mat(projectedPt);

				cv::Vec3f pt3;

				pt3(0) = reprojectedPt_m.ptr<float>(0)[0];
				pt3(1) = reprojectedPt_m.ptr<float>(1)[0];
				pt3(2) = reprojectedPt_m.ptr<float>(2)[0];

				pointMap.mvPoints[y * depthMat.cols + x] = pt3;
				pointMap.mvPointLocations.push_back(std::pair<int, int>(x, y));

			}
			else{
			}
		}
	}
}

void read_points_pointcloud(const PointMap& pointMap, cv::Mat& frameData_mmPoints){
	for (int i = 0; i < pointMap.mvPointLocations.size(); ++i){
		std::pair<int, int> xy = pointMap.mvPointLocations[i];
		int x = xy.first;
		int y = xy.second;
		int j = y * pointMap.width + x;
		frameData_mmPoints.ptr<float>(0)[i] = pointMap.mvPoints[j](0);
		frameData_mmPoints.ptr<float>(1)[i] = pointMap.mvPoints[j](1);
		frameData_mmPoints.ptr<float>(2)[i] = pointMap.mvPoints[j](2);
		frameData_mmPoints.ptr<float>(3)[i] = 1;
	}
}

void read_points_2Dmap(const PointMap& pointMap, cv::Mat& _2Dmap){
	for (int i = 0; i < pointMap.mvPointLocations.size(); ++i){
		std::pair<int, int> xy = pointMap.mvPointLocations[i];
		int x = xy.first;
		int y = xy.second;
		int j = y * pointMap.width + x;

		_2Dmap.ptr<cv::Vec4f>(y)[x](0) = pointMap.mvPoints[j](0);
		_2Dmap.ptr<cv::Vec4f>(y)[x](1) = pointMap.mvPoints[j](1);
		_2Dmap.ptr<cv::Vec4f>(y)[x](2) = pointMap.mvPoints[j](2);
		_2Dmap.ptr<cv::Vec4f>(y)[x](3) = 1;
	}
}


void load_frames(const std::vector<std::string>& filepaths, std::vector<PointMap>& pointMaps, std::vector<FrameData>& frameDatas){

	cv::FileStorage fs;

	for (auto it = filepaths.begin(); it != filepaths.end(); ++it){
		fs.open(*it, cv::FileStorage::READ);

		if (!fs.isOpened()) continue;

		std::cout << "loading " << *it << std::endl;

		cv::Mat depthMat;
		fs["depth"] >> depthMat;

		float win_width, win_height, fovy;
		cv::Mat colorMat, camera_extrinsic, camera_intrinsic;
		fs["color"] >> colorMat;

		fs["camera_intrinsic"]["width"] >> win_width;
		fs["camera_intrinsic"]["height"] >> win_height;
		fs["camera_intrinsic"]["fovy"] >> fovy;

		SkeletonNodeHard root;
		fs["skeleton"] >> root;


		PointMap pointMap(win_width, win_height);
		camera_intrinsic = generate_camera_intrinsic(win_width, win_height, fovy);
		read_depth_image(depthMat, camera_intrinsic, pointMap);
		FrameData frameData(pointMap.mvPointLocations.size(), depthMat, colorMat, camera_intrinsic, root);

		fs["camera_extrinsic"] >> frameData.mmCameraPose;

		fs.release();
		//prepare frame data(i.e. calculate normals and matrixify points)
		cv::Mat mDisplayNormals(win_height, win_width, CV_32FC3, cv::Scalar(0, 0, 0));

		read_points_pointcloud(pointMap, frameData.mmPoints);


		pointMaps.push_back(pointMap);
		frameDatas.push_back(frameData);
	}
}

PointMap fill_image_points(const std::vector<cv::Vec3f>& pts, const cv::Mat& cameraMatrix, int width, int height, float voxel_size){
	PointMap ret(width, height);
	int numPts = pts.size();

	cv::Mat pts_m(4, numPts, CV_32F, cv::Scalar(1));
	{
		cv::Mat pts_t(numPts, 3, CV_32F, (float*)pts.data());
		cv::Mat pts_m_3 = pts_t.t();
		pts_m_3.copyTo(pts_m(cv::Range(0, 3), cv::Range(0, numPts)));
	}

	cv::Mat pt_widths;
	cv::Mat pt_heights;
	if (voxel_size > 0){
		cv::Mat voxel_row = voxel_size * cv::Mat::ones(1, numPts, CV_32F);
		pt_widths = cv::Mat::zeros(4, numPts, CV_32F);
		pt_heights = cv::Mat::zeros(4, numPts, CV_32F);
		voxel_row.copyTo(pt_widths(cv::Range(0, 1), cv::Range(0, numPts)));
		voxel_row.copyTo(pt_heights(cv::Range(1, 2), cv::Range(0, numPts)));
		cv::add(pts_m, pt_widths, pt_widths);
		cv::add(pts_m, pt_heights, pt_heights);
	}

	cv::Mat pts2D = projective_data_association(pts_m, cv::Mat::eye(4, 4, CV_32F), cameraMatrix);

	cv::Mat pts2D_widths; 
	cv::Mat pts2D_heights;

	if (voxel_size > 0){
		pts2D_widths = projective_data_association(pt_widths, cv::Mat::eye(4, 4, CV_32F), cameraMatrix);
		pts2D_heights = projective_data_association(pt_heights, cv::Mat::eye(4, 4, CV_32F), cameraMatrix);
	}

	for (int i = 0; i < pts2D.cols; ++i){
		int x = pts2D.ptr<float>(0)[i];
		int y = pts2D.ptr<float>(1)[i];

		if (voxel_size > 0)
		{
			int w = pts2D_widths.ptr<float>(0)[i];
			int h = pts2D_heights.ptr<float>(1)[i];

			int min_x = std::min(x, w);
			int max_x = std::max(x, w);
			int min_y = std::min(y, h);
			int max_y = std::max(y, h);

			for (int _x = min_x; _x < max_x; ++_x){
				for (int _y = min_y; _y < max_y; ++_y){
					ret.addPoint(_x, _y, pts[i]);
				}
			}
		}
		else{
			ret.addPoint(x, y, pts[i]);
		}
	}
	return ret;
}


cv::Mat projective_data_association(const cv::Mat& C, const cv::Mat& targetTransform, const cv::Mat& cameraMatrix){
	int numPts = C.cols;
	cv::Mat projectedPts = cameraMatrix * targetTransform.inv() * C;
	cv::divide(projectedPts(cv::Range(0, 1), cv::Range(0, numPts)), projectedPts(cv::Range(2, 3), cv::Range(0, numPts)), projectedPts(cv::Range(0, 1), cv::Range(0, numPts)));
	cv::divide(projectedPts(cv::Range(1, 2), cv::Range(0, numPts)), projectedPts(cv::Range(2, 3), cv::Range(0, numPts)), projectedPts(cv::Range(1, 2), cv::Range(0, numPts)));

	return projectedPts(cv::Range(0, 2), cv::Range(0, numPts));
}


cv::Mat reproject_depth(const cv::Mat& projectedPts, const cv::Mat& targetDepthMat, const cv::Mat& cameraMatrix){

	int numPts = projectedPts.cols;

	cv::Mat cameraMatrix_inv = cameraMatrix.inv();

	cv::Mat depthValues(4, numPts, CV_32F, cv::Scalar(0));

	for (int i = 0; i < numPts; ++i){
		int x = projectedPts.ptr<float>(0)[i];
		int y = projectedPts.ptr<float>(1)[i];

		if (0 <= x && x < targetDepthMat.cols && 0 <= y && y < targetDepthMat.rows){
			float depth = targetDepthMat.ptr<float>(y)[x];
			depthValues.ptr<float>(0)[i] = depth;
			depthValues.ptr<float>(1)[i] = depth;
			depthValues.ptr<float>(2)[i] = depth;
			depthValues.ptr<float>(3)[i] = 1;
		}
	}

	cv::Mat screenProject(4, numPts, CV_32F, cv::Scalar(1));
	projectedPts.copyTo(screenProject(cv::Range(0, 2), cv::Range(0, numPts)));

	cv::Mat reprojectedPts;

	cv::multiply(depthValues, cameraMatrix_inv * screenProject, reprojectedPts);


	return reprojectedPts;
}
