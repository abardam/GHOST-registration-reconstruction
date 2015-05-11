#include <fstream>

#include <opencv2\opencv.hpp>
#include <cv_skeleton.h>
#include "skeletonconstraints.h"

//from assimp
#define AI_DEG_TO_RAD(x) ((x)*0.0174532925f)

SkeletonNodeHard generateFromReference(const SkeletonNodeHard * const ref, const SkeletonNodeHard * const prev){
	SkeletonNodeHard snh;
	cv::Mat generatedTransformation = skeleton_constraints_optimize(prev->mTransformation, ref->mTransformation, 1, 1);

	//cv::Mat rotationMatrix = generatedTransformation(cv::Range(0, 3), cv::Range(0, 3)) * prev->mTransformation(cv::Range(0, 3), cv::Range(0, 3));
	//cv::Mat translationMatrix = generatedTransformation(cv::Range(0, 3), cv::Range(3, 4)) + prev->mTransformation(cv::Range(0, 3), cv::Range(3, 4));
	//
	//rotationMatrix.copyTo(snh.mTransformation(cv::Range(0, 3), cv::Range(0, 3)));
	//translationMatrix.copyTo(snh.mTransformation(cv::Range(0, 3), cv::Range(3, 4)));


	for (int i = 0; i < prev->mChildren.size();++i){
		snh.mChildren.push_back(generateFromReference(&ref->mChildren[i], &prev->mChildren[i]));
	}

	snh.mTransformation = generatedTransformation * prev->mTransformation;

	//SVD correct scale error?
	//cv::SVD genSVD(snh.mTransformation);
	//cv::SVD refSVD(ref->mTransformation);
	//snh.mTransformation = genSVD.u * cv::Mat::diag(refSVD.w) * (genSVD.u * cv::Mat::diag(genSVD.w)).inv() * snh.mTransformation;

	//extract scale and correct it?
	cv::Vec3f scale(
		1/cv::norm(snh.mTransformation(cv::Range(0, 1), cv::Range(0, 3))),
		1/cv::norm(snh.mTransformation(cv::Range(1, 2), cv::Range(0, 3))),
		1/cv::norm(snh.mTransformation(cv::Range(2, 3), cv::Range(0, 3)))
		);

	cv::Mat rot = cv::Mat::diag(cv::Mat(scale)) * snh.mTransformation(cv::Range(0, 3), cv::Range(0, 3));
	rot.copyTo(snh.mTransformation(cv::Range(0, 3), cv::Range(0, 3)));

	snh.mName = prev->mName;
	snh.mParentName = prev->mParentName;

	return snh;
}

int main(int argc, char * argv[]){
	if (argc <= 1){
		std::cout << "Please enter directory\n";
		return 0;
	}

	std::string video_directory(argv[1]);
	int i = 0;
	std::stringstream filenameSS;
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

	//debug
	std::ofstream logstream_prev;
	logstream_prev.open("transform_log_gen.txt", std::ofstream::out);
	std::ofstream logstream_ref;
	logstream_ref.open("transform_log_ref.txt", std::ofstream::out);
	std::ofstream logstream_svd;
	logstream_svd.open("transform_log_svd.txt", std::ofstream::out);

	cv::Mat colorMat, camera_extrinsic, camera_intrinsic;
	SkeletonNodeHardMap snhMap;

	SkeletonNodeHard root;
	SkeletonNodeHard prevRoot;

	while (true){
		filenameSS.str("");
		filenameSS << video_directory << "/" << i << ".xml.gz";
		fs.open(filenameSS.str(), cv::FileStorage::READ);

		if (!fs.isOpened()) {
			break;
		}

		fs["color"] >> colorMat;
		fs["camera_extrinsic"] >> camera_extrinsic;
		//fs["camera_intrinsic"] >> camera_intrinsic;

		if (!fs["camera_intrinsic"].empty()){

			float win_width, win_height;
			float fovy;

			fs["camera_intrinsic"]["width"] >> win_width;
			fs["camera_intrinsic"]["height"] >> win_height;
			fs["camera_intrinsic"]["fovy"] >> fovy;

			camera_intrinsic = cv::Mat::eye(4, 4, CV_32F);
			camera_intrinsic.ptr<float>(0)[0] = -win_width / (2 * tan(AI_DEG_TO_RAD((fovy * (win_width / win_height) / 2.)))); //for some strange reason this is inaccurate for non-square aspect ratios
			camera_intrinsic.ptr<float>(1)[1] = win_height / (2 * tan(AI_DEG_TO_RAD(fovy / 2.)));
			camera_intrinsic.ptr<float>(0)[2] = win_width / 2 + 0.5;
			camera_intrinsic.ptr<float>(1)[2] = win_height / 2 + 0.5;
			//camera_intrinsic.ptr<float>(2)[2] = -1;
		}
		else{
			fs["camera_intrinsic_mat"] >> camera_intrinsic;
		}

		fs["skeleton"] >> root;

		SkeletonNodeHard gen_root;
		if (i>0){
			for (int i = 0; i < 1; ++i){
				gen_root = generateFromReference(&root, &prevRoot);
				prevRoot = gen_root;
			}
		}
		else{
			gen_root = root;
		}
		
		logstream_prev << "Frame " << i << std::endl << gen_root << std::endl;
		logstream_ref << "Frame " << i << std::endl << root << std::endl;

		cv::SVD prevSVD(gen_root.mChildren[0].mChildren[0].mTransformation);
		cv::SVD refSVD(root.mChildren[0].mChildren[0].mTransformation);

		logstream_svd << "Frame " << i << std::endl <<
			"Prev: " << std::endl << prevSVD.w << std::endl <<
			"Ref: " << std::endl << refSVD.w << std::endl;

		cv::Mat depth;
		fs["depth"] >> depth;

		cv::Mat depth_color(depth.rows, depth.cols, CV_8UC3);
		for (int i = 0; i < depth.rows*depth.cols; ++i){
			unsigned short z = depth.ptr<float>()[i] * 1000;
			depth_color.ptr<cv::Vec3b>()[i] = cv::Vec3b(z % 256, (z / 256) % 256, 0xff);
		}

		cv_draw_and_build_skeleton(&gen_root, cv::Mat::eye(4,4,CV_32F), camera_intrinsic, camera_extrinsic, &snhMap, depth_color); //change this to colorMat
		//for (auto it = bpdv.begin(); it != bpdv.end(); ++it){
		//	cv_draw_volume(*it, colorMat, camera_extrinsic, camera_intrinsic, snhMap);
		//}
		snhMap.clear();

		cv::imshow("color", depth_color); //change this to colorMat

		//cv::Mat depthMat;
		//fs["depth"] >> depthMat;
		//
		//cv::Mat depthHSV = depth_to_HSV(depthMat);
		//
		//cv::imshow("depth", depthHSV);

		cv::waitKey(1);
		++i;

		prevRoot = gen_root;
	}
	logstream_prev.close();
	logstream_ref.close();
	logstream_svd.close();
}