#include "point_to_plane.h"
#include "recons_common.h"
#include <cv_pointmat_common.h>
#include <cv_draw_common.h>

#define MAXIMUM_DEPTH_DIFFERENCE 0.1

void calculate_normals(const PointMap& depth_pointmap, const cv::Mat& input_points_2D, cv::Mat& normals, cv::Mat& mDisplayNormals){

	cv::Vec3f half(0.5, 0.5, 0.5);

	for (int i = 0; i < input_points_2D.cols; ++i){

		int x = input_points_2D.ptr<float>(0)[i];
		int y = input_points_2D.ptr<float>(1)[i];

		if (!CLAMP(x, y, depth_pointmap.width, depth_pointmap.height)) {
			std::cout << "Something went wrong (calculate_normals; invalid x/y)\n";
			continue;
		}
		
		const cv::Vec3f pt = depth_pointmap.mvPoints[y*depth_pointmap.width + x];

		//if (pt(2) < 0){ //TODO. Figure this shit out
		//if (pt(2) > 0){
		if (pt(2) != 0){
			cv::Vec3f v(0, 0, 0);

			const cv::Vec3f pt1y = depth_pointmap.mvPoints[(y + 1)*depth_pointmap.width + x];
			const cv::Vec3f pt1x = depth_pointmap.mvPoints[(y)*depth_pointmap.width + x + 1];

			

			//if (pt1y(2) < 0 && pt1x(2) < 0){ //TODO. Figure this shit out
			if (pt1y(2) != 0 && pt1x(2) != 0){
				v = (pt1y - pt).cross(pt1x - pt);

				if (v(2) == 0){
					normals.ptr<float>(0)[i] = 0;
					normals.ptr<float>(1)[i] = 0;
					normals.ptr<float>(2)[i] = 0;
					normals.ptr<float>(3)[i] = 1;
					continue;
				}

				v /= v(2);
				v = cv::normalize(v);
			}

			if (!mDisplayNormals.empty()){
				mDisplayNormals.ptr<cv::Vec3f>(y)[x] = v / 2 + half;
			}

			normals.ptr<float>(0)[i] = v(0);
			normals.ptr<float>(1)[i] = v(1);
			normals.ptr<float>(2)[i] = v(2);
			normals.ptr<float>(3)[i] = 1;
		}

	}
}

void point_to_plane_registration(
	const cv::Mat& source_pointmat,
	const cv::Mat& source_depth,
	const cv::Mat& source_cameramatrix,
	const cv::Mat& source_camerapose_inv,
	const cv::Mat& target_depth,
	const cv::Mat& target_cameramatrix,
	const cv::Mat& target_camerapose_inv,
	float voxel_size,
	cv::Mat& A, cv::Mat& b,
	bool verbose){

	//source
	cv::Mat C = source_pointmat.clone();
	//cv::Mat C1 = current.clone();

	float source_height = source_depth.rows,
		source_width = source_depth.cols;


	//get depth mat pointmap

	PointMap depth_pointmap(source_depth.cols, source_depth.rows);
	read_depth_image(source_depth, source_cameramatrix, depth_pointmap);
	cv::Mat depth_pointmat(4, depth_pointmap.mvPointLocations.size(), CV_32F);
	read_points_pointcloud(depth_pointmap, depth_pointmat);

	//lets try not using voxels
	//it works ok for the whole body but not for individual bps
	//cv::Mat C = depth_pointmat;

	//iterative part
	while (true)
	{

		const cv::Mat C_2D = projective_data_association(C, cv::Mat::eye(4, 4, CV_32F), source_cameramatrix);

		cv::Mat D = reproject_depth(C_2D, target_depth, target_cameramatrix);

		C = reproject_depth(C_2D, source_depth, source_cameramatrix); // lets try this? trip report: its bad UPDATE: actually its ok

		cv::Mat display_normals(source_depth.rows, source_depth.cols, CV_32FC3, cv::Scalar(0, 0, 0));

		cv::Mat normals(4, C.cols, CV_32F);
		calculate_normals(depth_pointmap, C_2D, normals, display_normals);

		cv::imshow("normals", display_normals);
		cv::waitKey(10);

		int point_to_plane_matches = 0;

		cv::Mat C_n;
		cv::Mat D_n;
		cv::Mat N_n;

		for (int i = 0; i < C.cols; ++i){
			//if (D.ptr<float>(2)[i] < 0  //TODO. Figure this shit out
			//if (D.ptr<float>(2)[i] > 0 
			if (D.ptr<float>(2)[i] != 0
				&& C.ptr<float>(2)[i] != 0
				&& abs(D.ptr<float>(2)[i] - C.ptr<float>(2)[i]) < MAXIMUM_DEPTH_DIFFERENCE //band-aid solution for wild depth differences; TODO. fix
				&& abs(source_pointmat.ptr<float>(2)[i] - C.ptr<float>(2)[i]) < MAXIMUM_DEPTH_DIFFERENCE //band-aid solution for wild depth differences; TODO. fix
				&& normals.ptr<float>(3)[i] == 1
				&& !(
				normals.ptr<float>(0)[i] == 0 &&
				normals.ptr<float>(1)[i] == 0 &&
				normals.ptr<float>(2)[i] == 0)
				){
				++point_to_plane_matches;
				C_n.push_back(C.col(i));
				D_n.push_back(D.col(i));
				N_n.push_back(normals.col(i));
			}
			else{
			}
		}

		if (point_to_plane_matches == 0){
			A = cv::Mat::zeros(6, 6, CV_64F);
			b = cv::Mat::zeros(6, 1, CV_64F);
			return;
		}

		if(verbose) std::cout << "point to plane matches (projective): " << point_to_plane_matches << std::endl;

		C_n = C_n.reshape(0, point_to_plane_matches).t();
		D_n = D_n.reshape(0, point_to_plane_matches).t();
		N_n = N_n.reshape(0, point_to_plane_matches).t();

		float energy;

		cv::Mat zerow = cv::Mat::zeros(1, N_n.cols, CV_32F);
		zerow.copyTo(N_n(cv::Range(3, 4), cv::Range(0, N_n.cols)));
		
		cv::Mat transformed_C = source_camerapose_inv * C_n;
		cv::Mat transformed_D = target_camerapose_inv * D_n;
		cv::Mat transformed_N = source_camerapose_inv * N_n;

		//cv::Mat transformed_C = C_n;
		//cv::Mat transformed_D = D_n;
		//cv::Mat transformed_N = N_n;

		//{
		//	std::cout << "debug ptmat draw AFTER TRANSFORMING; green is transformed_C, red is Transformed_D\n";
		//	std::vector<cv::Mat> ptmats;
		//	ptmats.push_back(transformed_C);
		//	ptmats.push_back(transformed_D);
		//	std::vector<cv::Vec3b> colors;
		//	colors.push_back(cv::Vec3b(0, 0xff, 0));
		//	colors.push_back(cv::Vec3b(0, 0, 0xff));
		//	display_pointmat("debug ptmat", target_depth.cols, target_depth.rows, target_cameramatrix, cv::Mat::eye(4, 4, CV_32F), ptmats, colors);
		//}

		for (int i = 0; i < transformed_N.cols; ++i){
			cv::Vec4f n = transformed_N.col(i);
			n = normalize(n);
			transformed_N.ptr<float>(0)[i] = n(0);
			transformed_N.ptr<float>(1)[i] = n(1);
			transformed_N.ptr<float>(2)[i] = n(2);
			transformed_N.ptr<float>(3)[i] = 0;
		}

		point_to_plane_linear(transformed_C, transformed_D, transformed_N, A, b);

		//{
		//
		//	cv::Mat x;
		//	cv::solve(A, b, x, cv::DECOMP_CHOLESKY);
		//	cv::Mat transformDelta = cv::Mat::eye(4, 4, CV_32F);
		//
		//	transformDelta.ptr<float>(0)[1] = x.ptr<float>(3)[0];
		//	transformDelta.ptr<float>(1)[0] = -x.ptr<float>(3)[0];
		//	transformDelta.ptr<float>(0)[2] = -x.ptr<float>(5)[0];
		//	transformDelta.ptr<float>(2)[0] = x.ptr<float>(5)[0];
		//	transformDelta.ptr<float>(1)[2] = x.ptr<float>(4)[0];
		//	transformDelta.ptr<float>(2)[1] = -x.ptr<float>(4)[0];
		//	transformDelta.ptr<float>(0)[3] = x.ptr<float>(0)[0];
		//	transformDelta.ptr<float>(1)[3] = x.ptr<float>(1)[0];
		//	transformDelta.ptr<float>(2)[3] = x.ptr<float>(2)[0];
		//
		//	std::cout << "debug ptmat draw AFTER APPLYING THE TRANSFORMATION; red is transformed_D, green is transformDelta * transformed_C\n";
		//	std::vector<cv::Mat> ptmats;
		//	ptmats.push_back(transformDelta * transformed_C);
		//	ptmats.push_back(transformed_D);
		//	std::vector<cv::Vec3b> colors;
		//	colors.push_back(cv::Vec3b(0, 0xff, 0));
		//	colors.push_back(cv::Vec3b(0, 0, 0xff));
		//	display_pointmat("debug ptmat", target_depth.cols, target_depth.rows, target_cameramatrix, cv::Mat::eye(4, 4, CV_32F), ptmats, colors);
		//}

		
		break;
	}
}

void point_to_plane_linear(const cv::Mat& C, const cv::Mat& D, const cv::Mat& N, cv::Mat& A, cv::Mat& b){

	A.create(6, 6, CV_32F);
	b.create(6, 1, CV_32F);


	cv::Mat C2N1, C1N2, C3N2, C2N3, C1N3, C3N1, C2N1_C1N2, C3N2_C2N3, C1N3_C3N1, C4N1, C4N2, C4N3, C1N1, C2N2, C3N3, D1N1, D2N2, D3N3, rest;
	cv::Mat a_coeff, b_coeff, g_coeff, tx_coeff, ty_coeff, tz_coeff, rest_coeff;
	float sum_a_coeff, sum_b_coeff, sum_g_coeff, sum_tx_coeff, sum_ty_coeff, sum_tz_coeff, sum_rest_coeff;

	vecMul(C, N, 1, 0, C2N1);
	vecMul(C, N, 0, 1, C1N2);
	vecMul(C, N, 2, 1, C3N2);
	vecMul(C, N, 1, 2, C2N3);
	vecMul(C, N, 2, 0, C3N1);
	vecMul(C, N, 0, 2, C1N3);
	vecMul(C, N, 3, 0, C4N1);
	vecMul(C, N, 3, 1, C4N2);
	vecMul(C, N, 3, 2, C4N3);
	vecMul(C, N, 0, 0, C1N1);
	vecMul(C, N, 1, 1, C2N2);
	vecMul(C, N, 2, 2, C3N3);
	vecMul(D, N, 0, 0, D1N1);
	vecMul(D, N, 1, 1, D2N2);
	vecMul(D, N, 2, 2, D3N3);

	cv::subtract(C2N1, C1N2, C2N1_C1N2);
	cv::subtract(C3N2, C2N3, C3N2_C2N3);
	cv::subtract(C1N3, C3N1, C1N3_C3N1);

	cv::add(C1N1, C2N2, rest);
	cv::add(C3N3, rest, rest);
	cv::subtract(rest, D1N1, rest);
	cv::subtract(rest, D2N2, rest);
	cv::subtract(rest, D3N3, rest);

	//alpha
	cv::multiply(C2N1_C1N2, C2N1_C1N2, a_coeff);
	cv::multiply(C2N1_C1N2, C3N2_C2N3, b_coeff);
	cv::multiply(C2N1_C1N2, C1N3_C3N1, g_coeff);
	cv::multiply(C2N1_C1N2, C4N1, tx_coeff);
	cv::multiply(C2N1_C1N2, C4N2, ty_coeff);
	cv::multiply(C2N1_C1N2, C4N3, tz_coeff);
	cv::multiply(C2N1_C1N2, rest, rest_coeff);
	
	sum_a_coeff = cv::sum(a_coeff)(0);
	sum_b_coeff = cv::sum(b_coeff)(0);
	sum_g_coeff = cv::sum(g_coeff)(0);
	sum_tx_coeff = cv::sum(tx_coeff)(0);
	sum_ty_coeff = cv::sum(ty_coeff)(0);
	sum_tz_coeff = cv::sum(tz_coeff)(0);
	sum_rest_coeff = cv::sum(rest_coeff)(0);

	A.ptr<float>(3)[0] = A.ptr<float>(0)[3] = sum_tx_coeff;
	A.ptr<float>(3)[1] = A.ptr<float>(1)[3] = sum_ty_coeff;
	A.ptr<float>(3)[2] = A.ptr<float>(2)[3] = sum_tz_coeff;
	A.ptr<float>(3)[3] = sum_a_coeff;
	A.ptr<float>(3)[4] = A.ptr<float>(4)[3] = sum_b_coeff;
	A.ptr<float>(3)[5] = A.ptr<float>(5)[3] = sum_g_coeff;
	b.ptr<float>(3)[0] = -sum_rest_coeff;


	//beta
	//cv::multiply(C3N2_C2N3, C2N1_C1N2, a_coeff);
	cv::multiply(C3N2_C2N3, C3N2_C2N3, b_coeff);
	cv::multiply(C3N2_C2N3, C1N3_C3N1, g_coeff);
	cv::multiply(C3N2_C2N3, C4N1, tx_coeff);
	cv::multiply(C3N2_C2N3, C4N2, ty_coeff);
	cv::multiply(C3N2_C2N3, C4N3, tz_coeff);
	cv::multiply(C3N2_C2N3, rest, rest_coeff);

	//sum_a_coeff = cv::sum(a_coeff)(0);
	sum_b_coeff = cv::sum(b_coeff)(0);
	sum_g_coeff = cv::sum(g_coeff)(0);
	sum_tx_coeff = cv::sum(tx_coeff)(0);
	sum_ty_coeff = cv::sum(ty_coeff)(0);
	sum_tz_coeff = cv::sum(tz_coeff)(0);
	sum_rest_coeff = cv::sum(rest_coeff)(0);

	A.ptr<float>(4)[0] = A.ptr<float>(0)[4] = sum_tx_coeff;
	A.ptr<float>(4)[1] = A.ptr<float>(1)[4] = sum_ty_coeff;
	A.ptr<float>(4)[2] = A.ptr<float>(2)[4] = sum_tz_coeff;
	//A.ptr<float>(4)[3] = A.ptr<float>(3)[4] = sum_a_coeff;
	A.ptr<float>(4)[4] = sum_b_coeff;
	A.ptr<float>(4)[5] = A.ptr<float>(5)[4] = sum_g_coeff;
	b.ptr<float>(4)[0] = -sum_rest_coeff;

	//gamma
	//cv::multiply(C1N3_C3N1, C2N1_C1N2, a_coeff);
	//cv::multiply(C1N3_C3N1, C3N2_C2N3, b_coeff);
	cv::multiply(C1N3_C3N1, C1N3_C3N1, g_coeff);
	cv::multiply(C1N3_C3N1, C4N1, tx_coeff);
	cv::multiply(C1N3_C3N1, C4N2, ty_coeff);
	cv::multiply(C1N3_C3N1, C4N3, tz_coeff);
	cv::multiply(C1N3_C3N1, rest, rest_coeff);

	//sum_a_coeff = cv::sum(a_coeff)(0);
	//sum_b_coeff = cv::sum(b_coeff)(0);
	sum_g_coeff = cv::sum(g_coeff)(0);
	sum_tx_coeff = cv::sum(tx_coeff)(0);
	sum_ty_coeff = cv::sum(ty_coeff)(0);
	sum_tz_coeff = cv::sum(tz_coeff)(0);
	sum_rest_coeff = cv::sum(rest_coeff)(0);

	A.ptr<float>(5)[0] = A.ptr<float>(0)[5] = sum_tx_coeff;
	A.ptr<float>(5)[1] = A.ptr<float>(1)[5] = sum_ty_coeff;
	A.ptr<float>(5)[2] = A.ptr<float>(2)[5] = sum_tz_coeff;
	//A.ptr<float>(5)[3] = A.ptr<float>(3)[5] = sum_a_coeff;
	//A.ptr<float>(5)[4] = A.ptr<float>(4)[5] = sum_b_coeff;
	A.ptr<float>(5)[5] = sum_g_coeff;
	b.ptr<float>(5)[0] = -sum_rest_coeff;

	//tx
	//cv::multiply(C4N1, C2N1_C1N2, a_coeff);
	//cv::multiply(C4N1, C3N2_C2N3, b_coeff);
	//cv::multiply(C4N1, C1N3_C3N1, g_coeff);
	cv::multiply(C4N1, C4N1, tx_coeff);
	cv::multiply(C4N1, C4N2, ty_coeff);
	cv::multiply(C4N1, C4N3, tz_coeff);
	cv::multiply(C4N1, rest, rest_coeff);

	//sum_a_coeff = cv::sum(a_coeff)(0);
	//sum_b_coeff = cv::sum(b_coeff)(0);
	//sum_g_coeff = cv::sum(g_coeff)(0);
	sum_tx_coeff = cv::sum(tx_coeff)(0);
	sum_ty_coeff = cv::sum(ty_coeff)(0);
	sum_tz_coeff = cv::sum(tz_coeff)(0);
	sum_rest_coeff = cv::sum(rest_coeff)(0);

	A.ptr<float>(0)[0] = sum_tx_coeff;
	A.ptr<float>(0)[1] = A.ptr<float>(1)[0] = sum_ty_coeff;
	A.ptr<float>(0)[2] = A.ptr<float>(2)[0] = sum_tz_coeff;
	//A.ptr<float>(0)[3] = A.ptr<float>(3)[0] = sum_a_coeff;
	//A.ptr<float>(0)[4] = A.ptr<float>(4)[0] = sum_b_coeff;
	//A.ptr<float>(0)[5] = A.ptr<float>(5)[0] = sum_g_coeff;
	b.ptr<float>(0)[0] = -sum_rest_coeff;

	//ty
	//cv::multiply(C4N2, C2N1_C1N2, a_coeff);
	//cv::multiply(C4N2, C3N2_C2N3, b_coeff);
	//cv::multiply(C4N2, C1N3_C3N1, g_coeff);
	//cv::multiply(C4N2, C4N1, tx_coeff);
	cv::multiply(C4N2, C4N2, ty_coeff);
	cv::multiply(C4N2, C4N3, tz_coeff);
	cv::multiply(C4N2, rest, rest_coeff);

	//sum_a_coeff = cv::sum(a_coeff)(0);
	//sum_b_coeff = cv::sum(b_coeff)(0);
	//sum_g_coeff = cv::sum(g_coeff)(0);
	//sum_tx_coeff = cv::sum(tx_coeff)(0);
	sum_ty_coeff = cv::sum(ty_coeff)(0);
	sum_tz_coeff = cv::sum(tz_coeff)(0);
	sum_rest_coeff = cv::sum(rest_coeff)(0);

	//A.ptr<float>(1)[0] = A.ptr<float>(0)[1] = sum_tx_coeff;
	A.ptr<float>(1)[1] = sum_ty_coeff;
	A.ptr<float>(1)[2] = A.ptr<float>(2)[1] = sum_tz_coeff;
	//A.ptr<float>(1)[3] = A.ptr<float>(3)[1] = sum_a_coeff;
	//A.ptr<float>(1)[4] = A.ptr<float>(4)[1] = sum_b_coeff;
	//A.ptr<float>(1)[5] = A.ptr<float>(5)[1] = sum_g_coeff;
	b.ptr<float>(1)[0] = -sum_rest_coeff;

	//tz
	//cv::multiply(C4N3, C2N1_C1N2, a_coeff);
	//cv::multiply(C4N3, C3N2_C2N3, b_coeff);
	//cv::multiply(C4N3, C1N3_C3N1, g_coeff);
	//cv::multiply(C4N3, C4N1, tx_coeff);
	//cv::multiply(C4N3, C4N2, ty_coeff);
	cv::multiply(C4N3, C4N3, tz_coeff);
	cv::multiply(C4N3, rest, rest_coeff);

	//sum_a_coeff = cv::sum(a_coeff)(0);
	//sum_b_coeff = cv::sum(b_coeff)(0);
	//sum_g_coeff = cv::sum(g_coeff)(0);
	//sum_tx_coeff = cv::sum(tx_coeff)(0);
	//sum_ty_coeff = cv::sum(ty_coeff)(0);
	sum_tz_coeff = cv::sum(tz_coeff)(0);
	sum_rest_coeff = cv::sum(rest_coeff)(0);

	//A.ptr<float>(2)[0] = A.ptr<float>(0)[2] = sum_tx_coeff;
	//A.ptr<float>(2)[1] = A.ptr<float>(1)[2] = sum_ty_coeff;
	A.ptr<float>(2)[2] = sum_tz_coeff;
	//A.ptr<float>(2)[3] = A.ptr<float>(3)[2] = sum_a_coeff;
	//A.ptr<float>(2)[4] = A.ptr<float>(4)[2] = sum_b_coeff;
	//A.ptr<float>(2)[5] = A.ptr<float>(5)[2] = sum_g_coeff;
	b.ptr<float>(2)[0] = -sum_rest_coeff;
}


cv::Mat point_to_plane_optimize(const cv::Mat& C, const cv::Mat& D, const cv::Mat& N, float * energy, float weight){

	cv::Mat A, b, x;

	point_to_plane_linear(C, D, N, A, b);

	cv::solve(A, b, x, cv::DECOMP_CHOLESKY);
	cv::Mat transformDelta = cv::Mat::eye(4, 4, CV_32F);

	transformDelta.ptr<float>(0)[1] = weight * x.ptr<float>(3)[0];
	transformDelta.ptr<float>(1)[0] = weight * -x.ptr<float>(3)[0];
	transformDelta.ptr<float>(0)[2] = weight * -x.ptr<float>(5)[0];
	transformDelta.ptr<float>(2)[0] = weight * x.ptr<float>(5)[0];
	transformDelta.ptr<float>(1)[2] = weight * x.ptr<float>(4)[0];
	transformDelta.ptr<float>(2)[1] = weight * -x.ptr<float>(4)[0];
	transformDelta.ptr<float>(0)[3] = weight * x.ptr<float>(0)[0];
	transformDelta.ptr<float>(1)[3] = weight * x.ptr<float>(1)[0];
	transformDelta.ptr<float>(2)[3] = weight * x.ptr<float>(2)[0];

	if (energy != 0){
		*energy = cv::norm(A*x - b);
	}

	return transformDelta;
}