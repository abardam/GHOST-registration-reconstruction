#include <opencv2\opencv.hpp>
#include "recons_optimization.h"

#define SKELETON_CONSTRAINT_WEIGHT 1


int main(int argc, char** argv){

	float voxel_size = 0;

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
	int numframes = 50;
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

	std::vector<PointMap> pointMaps;
	std::vector<FrameData> frameDatas;

	load_frames(filenames, pointMaps, frameDatas);

	//first frame init
	SkeletonNodeHard gen_root = frameDatas[0].mmRoot;

	while (true){

		cv::Mat current;
		cv::Mat currentTransform = cv::Mat::eye(4, 4, CV_32F);

		for (int frame = 0; frame < numframes - 1; ++frame){

			while (true)
			{

				current = currentTransform * frameDatas[0].mmPoints;

				int sourceFrame = frame;
				int targetFrame = frame + 1;

				//current = frameDatas[sourceFrame].mmPoints;

				cv::Mat mDisplay(pointMaps[frame].height, pointMaps[frame].width, CV_8UC4, cv::Scalar(0, 0, 0, 0xff));

				cv::Mat A = cv::Mat::zeros(6, 6, CV_32F);
				cv::Mat b = cv::Mat::zeros(6, 1, CV_32F);

				//point to point registration
				if (point_to_point)
				{

					cv::Mat _A, _b;

					point_to_point_registration(frameDatas[sourceFrame].mmPoints,
						frameDatas[sourceFrame].mmColor,
						frameDatas[sourceFrame].mmCameraMatrix,
						cv::Mat::eye(4,4,CV_32F),
						frameDatas[targetFrame].mmColor,
						frameDatas[targetFrame].mmDepth,
						frameDatas[targetFrame].mmCameraMatrix,
						_A, _b);

					cv::add(_A, A, A);
					cv::add(_b, b, b);
				}


				//point to plane registration
				if (point_to_plane)
				{

					cv::Mat _A, _b;

					point_to_plane_registration(
						frameDatas[sourceFrame].mmPoints,
						frameDatas[sourceFrame].mmDepth,
						frameDatas[sourceFrame].mmCameraMatrix,
						cv::Mat::eye(4,4,CV_32F),
						frameDatas[targetFrame].mmDepth,
						frameDatas[targetFrame].mmCameraMatrix,
						cv::Mat::eye(4, 4, CV_32F),
						voxel_size,
						_A, _b);
					cv::add(_A, A, A);
					cv::add(_b, b, b);


				}

				if (skeleton){
					cv::Mat _A, _b;
					//skeleton_constraints_linear(frameDatas[sourceFrame].mmRoot.mChildren[0].mTransformation, frameDatas[targetFrame].mmRoot.mChildren[0].mTransformation, 1, 1, _A, _b);
					skeleton_constraints_linear(frameDatas[sourceFrame].mmCameraPose,
						frameDatas[targetFrame].mmCameraPose, 1, 1, _A, _b);

					cv::add(SKELETON_CONSTRAINT_WEIGHT*_A, A, A);
					cv::add(SKELETON_CONSTRAINT_WEIGHT*_b, b, b);
				}


				//display points

				draw_points_on_image(frameDatas[sourceFrame].mmPoints, frameDatas[sourceFrame].mmCameraMatrix, mDisplay, cv::Vec4b(0xff, 0, 0, 0xff));

				//cv::imshow("display", mDisplay);
				//cv::waitKey();

				draw_points_on_image(frameDatas[targetFrame].mmPoints, frameDatas[targetFrame].mmCameraMatrix, mDisplay, cv::Vec4b(0, 0xff, 0, 0xff));


				//cv::imshow("display", mDisplay);
				//cv::waitKey();


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

				cv::Mat E = transformDelta * current;

				//cv::Mat difference = E - frameDatas[frame-1].mmPoints;
				//
				//float nrm = 0;
				//for (int i = 0; i < E.cols; ++i){
				//	nrm += cv::norm(difference(cv::Range(0, 3), cv::Range(i, i + 1)));
				//}

				//std::cout << "energy: " << energy << std::endl;
				//<< "SAD: " << nrm << std::endl;

				draw_points_on_image(E, frameDatas[sourceFrame].mmCameraMatrix, mDisplay, cv::Vec4b(0, 0, 0xff, 0xff));

				currentTransform = transformDelta * currentTransform;

				std::cout << "difference: " << std::endl << transformDelta *  frameDatas[sourceFrame].mmCameraPose - frameDatas[targetFrame].mmCameraPose << std::endl;


				cv::imshow("display", mDisplay);
				char inp = cv::waitKey();

				break;
			}
		}

	}
}