#include <fstream>

#include <opencv2\opencv.hpp>
#include <cv_skeleton.h>
#include "skeletonconstraints.h"

//from assimp
#define AI_DEG_TO_RAD(x) ((x)*0.0174532925f)


SkeletonNodeHard cmpSNH(const SkeletonNodeHard& a, const SkeletonNodeHard& b){
	SkeletonNodeHard snh;
	snh.mTransformation = a.mTransformation - b.mTransformation;
	for (int i = 0; i < a.mChildren.size(); ++i){
		snh.mChildren.push_back(cmpSNH(a.mChildren[i], b.mChildren[i]));
	}
	snh.mName = a.mName;
	snh.mParentName = a.mParentName;
	return snh;
}


SkeletonNodeHard generateFromReference(const SkeletonNodeHard * const ref, const SkeletonNodeHard * const prev){
	SkeletonNodeHard snh;
	snh.mTransformation = prev->mTransformation.clone();

	float difference;

	do{
		cv::Mat generatedTransformation = skeleton_constraints_optimize(snh.mTransformation, ref->mTransformation, 1, 1);
		snh.mTransformation = generatedTransformation * snh.mTransformation;

		cv::Mat cmpmat = generatedTransformation - cv::Mat::eye(4, 4, CV_32F);
		difference = cv::norm(cmpmat.col(0)) + cv::norm(cmpmat.col(1)) + cv::norm(cmpmat.col(2)) + cv::norm(cmpmat.col(3));

	} while (difference > 0.00001);
	//cv::Mat rotationMatrix = generatedTransformation(cv::Range(0, 3), cv::Range(0, 3)) * prev->mTransformation(cv::Range(0, 3), cv::Range(0, 3));
	//cv::Mat translationMatrix = generatedTransformation(cv::Range(0, 3), cv::Range(3, 4)) + prev->mTransformation(cv::Range(0, 3), cv::Range(3, 4));
	//
	//rotationMatrix.copyTo(snh.mTransformation(cv::Range(0, 3), cv::Range(0, 3)));
	//translationMatrix.copyTo(snh.mTransformation(cv::Range(0, 3), cv::Range(3, 4)));


	for (int i = 0; i < prev->mChildren.size();++i){
		snh.mChildren.push_back(generateFromReference(&ref->mChildren[i], &prev->mChildren[i]));
	}


	//SVD correct scale error?
	//cv::SVD genSVD(snh.mTransformation);
	//cv::SVD refSVD(ref->mTransformation);
	//snh.mTransformation = genSVD.u * cv::Mat::diag(refSVD.w) * (genSVD.u * cv::Mat::diag(genSVD.w)).inv() * snh.mTransformation;

	//extract scale and correct it?
	cv::Vec3f scale(
		cv::norm(ref->mTransformation(cv::Range(0, 1), cv::Range(0, 3))) / cv::norm(snh.mTransformation(cv::Range(0, 1), cv::Range(0, 3))),
		cv::norm(ref->mTransformation(cv::Range(1, 2), cv::Range(0, 3))) / cv::norm(snh.mTransformation(cv::Range(1, 2), cv::Range(0, 3))),
		cv::norm(ref->mTransformation(cv::Range(2, 3), cv::Range(0, 3))) / cv::norm(snh.mTransformation(cv::Range(2, 3), cv::Range(0, 3)))
		);

	cv::Mat rot = cv::Mat::diag(cv::Mat(scale)) * snh.mTransformation(cv::Range(0, 3), cv::Range(0, 3));
	rot.copyTo(snh.mTransformation(cv::Range(0, 3), cv::Range(0, 3)));

	snh.mName = prev->mName;
	snh.mParentName = prev->mParentName;

	return snh;
}



SkeletonNodeAbsoluteVector generateFromReference(const SkeletonNodeAbsoluteVector& ref, const SkeletonNodeAbsoluteVector& prev){

	SkeletonNodeAbsoluteVector snav(ref.size());

	for (int i = 0; i < ref.size(); ++i){
		SkeletonNodeHard snh;
		snh.mTransformation = prev[i].mTransformation.clone();

		float difference;

		do{
			cv::Mat generatedTransformation = skeleton_constraints_optimize(snh.mTransformation, ref[i].mTransformation, 1, 1);
			snh.mTransformation = generatedTransformation * snh.mTransformation;

			cv::Mat cmpmat = generatedTransformation - cv::Mat::eye(4, 4, CV_32F);
			difference = cv::norm(cmpmat.col(0)) + cv::norm(cmpmat.col(1)) + cv::norm(cmpmat.col(2)) + cv::norm(cmpmat.col(3));

		} while (difference > 0.00001);


		//extract scale and correct it?
		cv::Vec3f scale(
			cv::norm(ref[i].mTransformation(cv::Range(0, 1), cv::Range(0, 3))) / cv::norm(snh.mTransformation(cv::Range(0, 1), cv::Range(0, 3))),
			cv::norm(ref[i].mTransformation(cv::Range(1, 2), cv::Range(0, 3))) / cv::norm(snh.mTransformation(cv::Range(1, 2), cv::Range(0, 3))),
			cv::norm(ref[i].mTransformation(cv::Range(2, 3), cv::Range(0, 3))) / cv::norm(snh.mTransformation(cv::Range(2, 3), cv::Range(0, 3)))
			);

		cv::Mat rot = cv::Mat::diag(cv::Mat(scale)) * snh.mTransformation(cv::Range(0, 3), cv::Range(0, 3));
		rot.copyTo(snh.mTransformation(cv::Range(0, 3), cv::Range(0, 3)));

		snh.mName = prev[i].mName;
		snh.mParentName = prev[i].mParentName;
		snav[i] = snh;
	}

	return snav;
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
	std::ofstream logstream_cmp;
	logstream_cmp.open("transform_log_cmp.txt", std::ofstream::out);

	cv::Mat colorMat, camera_extrinsic, camera_intrinsic;
	SkeletonNodeHardMap snhMap;
	SkeletonNodeHardMap snhMap2;

	SkeletonNodeHard root;
	SkeletonNodeHard prevRoot;
	SkeletonNodeAbsoluteVector prevroot_absolute;

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

		std::vector<SkeletonNodeHard> root_absolute;
		absolutize_snh(root, root_absolute);


		SkeletonNodeAbsoluteVector gen_root;
		if (i>0){
			for (int i = 0; i < 1; ++i){
				gen_root = generateFromReference(root_absolute, prevroot_absolute);
				prevroot_absolute = gen_root;
			}
		}
		else{
			gen_root = root_absolute;
		}
		
		//logstream_prev << "Frame " << i << std::endl << gen_root << std::endl;
		//logstream_ref << "Frame " << i << std::endl << root << std::endl;
		//logstream_cmp << "Frame" << i << std::endl << cmpSNH(root, gen_root) << std::endl;
		//
		//cv::SVD prevSVD(gen_root.mChildren[0].mChildren[0].mTransformation);
		//cv::SVD refSVD(root.mChildren[0].mChildren[0].mTransformation);
		//
		//logstream_svd << "Frame " << i << std::endl <<
		//	"Prev: " << std::endl << prevSVD.w << std::endl <<
		//	"Ref: " << std::endl << refSVD.w << std::endl;

		cv::Mat depth;
		fs["depth"] >> depth;

		cv::Mat depth_color(depth.rows, depth.cols, CV_8UC3);
		for (int i = 0; i < depth.rows*depth.cols; ++i){
			unsigned short z = depth.ptr<float>()[i] * 1000;
			depth_color.ptr<cv::Vec3b>()[i] = cv::Vec3b(z % 256, (z / 256) % 256, 0xff);
		}

		for (int i = 0; i < gen_root.size(); ++i){

			cv::Mat zero_pt(cv::Vec4f(0, 0, 0, 1));
			cv::Mat x_pt(cv::Vec4f(0.1, 0, 0, 1));
			cv::Mat y_pt(cv::Vec4f(0, 0.1, 0, 1));
			cv::Mat z_pt(cv::Vec4f(0, 0, 0.1, 1));

			cv::Mat bpt = gen_root[i].mTransformation;

			cv::Mat zero_pt_trans = camera_intrinsic * bpt * zero_pt;
			cv::Mat x_pt_trans = camera_intrinsic * bpt * x_pt;
			cv::Mat y_pt_trans = camera_intrinsic * bpt * y_pt;
			cv::Mat z_pt_trans = camera_intrinsic * bpt * z_pt;

			cv::Point zero_pt_2d(zero_pt_trans.ptr<float>(0)[0] / zero_pt_trans.ptr<float>(2)[0], zero_pt_trans.ptr<float>(1)[0] / zero_pt_trans.ptr<float>(2)[0]);
			cv::Point x_pt_2d(y_pt_trans.ptr<float>(0)[0] / x_pt_trans.ptr<float>(2)[0], x_pt_trans.ptr<float>(1)[0] / x_pt_trans.ptr<float>(2)[0]);
			cv::Point y_pt_2d(y_pt_trans.ptr<float>(0)[0] / y_pt_trans.ptr<float>(2)[0], y_pt_trans.ptr<float>(1)[0] / y_pt_trans.ptr<float>(2)[0]);
			cv::Point z_pt_2d(y_pt_trans.ptr<float>(0)[0] / z_pt_trans.ptr<float>(2)[0], z_pt_trans.ptr<float>(1)[0] / z_pt_trans.ptr<float>(2)[0]);

			cv::line(colorMat, zero_pt_2d, x_pt_2d, cv::Scalar(0xff, 0, 0));
			cv::line(colorMat, zero_pt_2d, y_pt_2d, cv::Scalar(0, 0xff, 0));
			cv::line(colorMat, zero_pt_2d, z_pt_2d, cv::Scalar(0, 0, 0xff));
		}

		snhMap.clear();
		snhMap2.clear();

		cv::imshow("color", colorMat);

		//cv::Mat depthMat;
		//fs["depth"] >> depthMat;
		//
		//cv::Mat depthHSV = depth_to_HSV(depthMat);
		//
		//cv::imshow("depth", depthHSV);

		cv::waitKey(1);
		++i;

		prevroot_absolute = gen_root;
	}
	logstream_prev.close();
	logstream_ref.close();
	logstream_svd.close();
}