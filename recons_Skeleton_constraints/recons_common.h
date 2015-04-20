#pragma once

#include <opencv2\opencv.hpp>
#include <AssimpOpenGL.h>
#include <cv_skeleton.h>

#define vecMul(Mat, Mat2, n, m, outMat) cv::multiply(Mat(cv::Range(n,n+1), cv::Range(0, Mat.cols)), Mat2(cv::Range(m,m+1), cv::Range(0, Mat2.cols)), outMat)

//point map struct that represents a 2D map of 3D points
struct PointMap{
	int width, height;
	std::vector<cv::Vec3f> mvPoints; //vector of 3D points arranged according to 2D location
	std::vector<std::pair<int, int>> mvPointLocations; //list of 2D coordinates with valid 3D points

	PointMap(int w, int h);
	void PointMap::addPoint(int x, int y, const cv::Vec3f& pt);

};

//represents a single frame
struct FrameData{
	cv::Mat mmDepth;
	cv::Mat mmColor;
	cv::Mat mmPoints;
	cv::Mat mmNormals;
	cv::Mat mmCameraMatrix;
	cv::Mat mmCameraPose;
	SkeletonNodeHard mmRoot;
	int width, height;

	FrameData(int numPts, const cv::Mat& depthMat, const cv::Mat& colorMat, const cv::Mat& cameraMatrix, const SkeletonNodeHard& root);
	FrameData(const FrameData&);
};

//generates intrinsic camera parameters based on width, height, and y-field of view
cv::Mat generate_camera_intrinsic(int win_width, int win_height, float fovy);

//converts depth map to point map
void read_depth_image(const cv::Mat& depthMat, const cv::Mat& camera_intrinsic, PointMap& pointMap);

//reads point map data into a point cloud (4xN cv::Mat)
void read_points_pointcloud(const PointMap& pointMap, cv::Mat& frameData_mmPoints);

//reads point map data into a 2D map (WxHx4 cv::Mat)
void read_points_2Dmap(const PointMap& pointMap, cv::Mat& frameData_mmPoints);

//utility function to load in frames (output: point maps and frame datas)
void load_frames(const std::vector<std::string>& filepaths, std::vector<PointMap>& pointMaps, std::vector<FrameData>& frameDatas, bool load_depth = true, bool load_color = true);

PointMap fill_image_points(const std::vector<cv::Vec3f>& pts, const cv::Mat& cameraMatrix, int width, int height, float voxel_size);


//basically 3D to 2D
cv::Mat projective_data_association(const cv::Mat& C, const cv::Mat& targetTransform, const cv::Mat& cameraMatrix);

//basically 2D to 3D (need a depth mat)
cv::Mat reproject_depth(const cv::Mat& projectedPts, const cv::Mat& targetDepthMat, const cv::Mat& cameraMatrix);


template <typename T>
void draw_points_on_image(const cv::Mat& points, const cv::Mat& cameraMatrix, cv::Mat& image, T color){
	for (int i = 0; i < points.cols; ++i){
		float depth = points.ptr<float>(2)[i];
		cv::Mat projectedPt(4, 1, CV_32F);
		projectedPt.ptr<float>(0)[0] = points.ptr<float>(0)[i] / depth;
		projectedPt.ptr<float>(1)[0] = points.ptr<float>(1)[i] / depth;
		projectedPt.ptr<float>(2)[0] = 1;
		projectedPt.ptr<float>(3)[0] = 1;

		cv::Mat depthPt = cameraMatrix * projectedPt;

		int x = depthPt.ptr<float>(0)[0];
		int y = depthPt.ptr<float>(1)[0];

		if (x >= 0 && x < image.cols &&y >= 0 && y < image.rows)

			image.ptr<T>(y)[x] = color;
	}
}
