#include "point_to_point.h"
#include "recons_common.h"

void point_to_point_registration(
	const cv::Mat& source_pointmat, 
	const cv::Mat& source_color,
	const cv::Mat& source_depth,
	const cv::Mat& source_cameramatrix, 
	const cv::Mat& source_camerapose_inv,
	const cv::Mat& target_color, 
	const cv::Mat& target_depth, 
	const cv::Mat& target_cameramatrix,
	const cv::Mat& target_camerapose_inv,
	cv::Mat& A, cv::Mat& b,
	bool verbose){

	cv::Mat C = source_pointmat.clone();
	int N = C.cols;
	//cv::Mat C = current.clone();

	cv::Mat C_2D_t = projective_data_association(C, cv::Mat::eye(4, 4, CV_32F), source_cameramatrix).t();
	std::vector<unsigned char> status(N);
	std::vector<float> error(N);

	cv::Mat C_2D_r = C_2D_t.reshape(2, N);
	cv::Mat D_2D_r(1, N, CV_32FC2);

	cv::Mat sourceImage, targetImage;
	cv::cvtColor(source_color, sourceImage, CV_BGR2GRAY);
	cv::cvtColor(target_color, targetImage, CV_BGR2GRAY);

	cv::calcOpticalFlowPyrLK(sourceImage, targetImage, C_2D_r, D_2D_r, status, error);

	cv::Mat D_2D = D_2D_r.reshape(1, N).t();

	cv::Mat D = reproject_depth(D_2D, target_depth, target_cameramatrix);

	//debug
#if 0
	cv::Mat s(source_color.rows, source_color.cols, CV_8UC3, cv::Scalar(0, 0, 0));
	for (int i = 0; i < C.cols; ++i){
		float depth = C.ptr<float>(2)[i];
		cv::Mat projectedPt(4, 1, CV_32F);
		projectedPt.ptr<float>(0)[0] = C.ptr<float>(0)[i] / depth;
		projectedPt.ptr<float>(1)[0] = C.ptr<float>(1)[i] / depth;
		projectedPt.ptr<float>(2)[0] = 1;
		projectedPt.ptr<float>(3)[0] = 1;
	
		cv::Mat depthPt = source_cameramatrix * projectedPt;
	
		int x = depthPt.ptr<float>(0)[0];
		int y = depthPt.ptr<float>(1)[0];
	
		if (x >= 0 && x < source_color.cols&&y >= 0 && y < source_color.rows)
			s.ptr<cv::Vec3b>(y)[x] = cv::Vec3b(0xff, 0, 0);
	}
	
	
	cv::imshow("1", s);
	cv::waitKey();
	
	
	for (int i = 0; i < D.cols; ++i){
		float depth = D.ptr<float>(2)[i];
		cv::Mat projectedPt(4, 1, CV_32F);
		projectedPt.ptr<float>(0)[0] = D.ptr<float>(0)[i] / depth;
		projectedPt.ptr<float>(1)[0] = D.ptr<float>(1)[i] / depth;
		projectedPt.ptr<float>(2)[0] = 1;
		projectedPt.ptr<float>(3)[0] = 1;
	
		cv::Mat depthPt = target_cameramatrix * projectedPt;
	
		int x = depthPt.ptr<float>(0)[0];
		int y = depthPt.ptr<float>(1)[0];
	
		if (x >= 0 && x < source_color.cols&&y >= 0 && y < source_color.rows)
			s.ptr<cv::Vec3b>(y)[x] = cv::Vec3b(0, 0xff, 0);
	}
	
	
	cv::imshow("1", s);
	cv::waitKey();
#endif
	//debug
#if 0
	{
		s = source_color.clone();
		for (int i = 0; i < N; i += 1){
			cv::Point pt = C_2D_r.ptr<cv::Vec2f>()[i];
			cv::circle(s, pt, 1, cv::Scalar(0, 0xff, 0xff), -1);
		}
		cv::imshow("1", s);
		cv::waitKey();

		s = target_color.clone();
		for (int i = 0; i < N; i += 1){
			cv::Point pt = D_2D_r.ptr<cv::Vec2f>()[i];
			cv::circle(s, pt, 1, cv::Scalar(0xff, 0, 0xff), -1);
		}
		cv::imshow("1", s);
		cv::waitKey();
	}
#endif

	C = reproject_depth(C_2D_t.t(), source_depth, source_cameramatrix); // lets try this? trip report: its bad UPDATE: actually its ok
	int point_to_point_matches = 0;

	cv::Mat C_n;
	cv::Mat D_n;

	for (int i = 0; i < C.cols; ++i){
		//if (D.ptr<float>(2)[i] < 0 && //TODO. figure this shit out
		//if (D.ptr<float>(2)[i] > 0 &&
		if (D.ptr<float>(2)[i] != 0 &&
			C.ptr<float>(2)[i] != 0 &&
			status[i] == 1 &&
			error[i] < OPTICAL_FLOW_ERROR_THRESHOLD){
			++point_to_point_matches;
			C_n.push_back(C.col(i));
			D_n.push_back(D.col(i));
		}
		else{
		}
	}
	if(verbose) std::cout << "point to point matches (optical flow): " << point_to_point_matches << std::endl;

	if (point_to_point_matches < OPTICAL_FLOW_MIN_MATCHES){
		A = cv::Mat(6, 6, CV_32F, cv::Scalar(0));
		b = cv::Mat(6, 1, CV_32F, cv::Scalar(0));
		return;
	}

	C_n = C_n.reshape(0, point_to_point_matches).t();
	D_n = D_n.reshape(0, point_to_point_matches).t();
	point_to_point_linear(source_camerapose_inv * C_n, target_camerapose_inv * D_n, A, b);

}

void point_to_point_linear(const cv::Mat& C, const cv::Mat& D, cv::Mat& M, cv::Mat& b){
	M = cv::Mat::zeros(6, 6, CV_32F);

	cv::Mat C_4_sq;
	float sum_C_4_sq;
	vecMul(C, C, 3, 3, C_4_sq);
	sum_C_4_sq = cv::sum(C_4_sq)(0);

	M.ptr<float>(0)[0] = sum_C_4_sq;
	M.ptr<float>(1)[1] = sum_C_4_sq;
	M.ptr<float>(2)[2] = sum_C_4_sq;


	cv::Mat C_1_sq;
	float sum_C_1_sq;
	vecMul(C, C, 0, 0, C_1_sq);
	sum_C_1_sq = cv::sum(C_1_sq)(0);
	cv::Mat C_2_sq;
	float sum_C_2_sq;
	vecMul(C, C, 1, 1, C_2_sq);
	sum_C_2_sq = cv::sum(C_2_sq)(0);
	cv::Mat C_3_sq;
	float sum_C_3_sq;
	vecMul(C, C, 2, 2, C_3_sq);
	sum_C_3_sq = cv::sum(C_3_sq)(0);

	M.ptr<float>(3)[3] = sum_C_1_sq + sum_C_2_sq;
	M.ptr<float>(4)[4] = sum_C_2_sq + sum_C_3_sq;
	M.ptr<float>(5)[5] = sum_C_1_sq + sum_C_3_sq;

	cv::Mat C_2_C_4;
	float sum_C_2_C_4;
	vecMul(C, C, 1, 3, C_2_C_4);
	sum_C_2_C_4 = cv::sum(C_2_C_4)(0);

	cv::Mat C_3_C_4;
	float sum_C_3_C_4;
	vecMul(C, C, 2, 3, C_3_C_4);
	sum_C_3_C_4 = cv::sum(C_3_C_4)(0);

	cv::Mat C_1_C_4;
	float sum_C_1_C_4;
	vecMul(C, C, 0, 3, C_1_C_4);
	sum_C_1_C_4 = cv::sum(C_1_C_4)(0);

	cv::Mat C_1_C_3;
	float sum_C_1_C_3;
	vecMul(C, C, 0, 2, C_1_C_3);
	sum_C_1_C_3 = cv::sum(C_1_C_3)(0);

	cv::Mat C_2_C_3;
	float sum_C_2_C_3;
	vecMul(C, C, 1, 2, C_2_C_3);
	sum_C_2_C_3 = cv::sum(C_2_C_3)(0);

	cv::Mat C_1_C_2;
	float sum_C_1_C_2;
	vecMul(C, C, 0, 1, C_1_C_2);
	sum_C_1_C_2 = cv::sum(C_1_C_2)(0);

	M.ptr<float>(0)[3] = M.ptr<float>(3)[0] = sum_C_2_C_4;
	M.ptr<float>(0)[5] = M.ptr<float>(5)[0] = -sum_C_3_C_4;
	M.ptr<float>(1)[3] = M.ptr<float>(3)[1] = -sum_C_1_C_4;
	M.ptr<float>(1)[4] = M.ptr<float>(4)[1] = sum_C_3_C_4;
	M.ptr<float>(2)[4] = M.ptr<float>(4)[2] = -sum_C_2_C_4;
	M.ptr<float>(2)[5] = M.ptr<float>(5)[2] = sum_C_1_C_4;
	M.ptr<float>(3)[4] = M.ptr<float>(4)[3] = -sum_C_1_C_3;
	M.ptr<float>(3)[5] = M.ptr<float>(5)[3] = -sum_C_2_C_3;
	M.ptr<float>(4)[5] = M.ptr<float>(5)[4] = -sum_C_1_C_2;
	
	cv::Mat D_1_C_4;
	vecMul(D, C, 0, 3, D_1_C_4);
	float sum_D_1_C_4 = cv::sum(D_1_C_4)(0);

	cv::Mat D_2_C_4;
	vecMul(D, C, 1, 3, D_2_C_4);
	float sum_D_2_C_4 = cv::sum(D_2_C_4)(0);

	cv::Mat D_3_C_4;
	vecMul(D, C, 2, 3, D_3_C_4);
	float sum_D_3_C_4 = cv::sum(D_3_C_4)(0);

	cv::Mat D_1_C_2;
	vecMul(D, C, 0, 1, D_1_C_2);
	float sum_D_1_C_2 = cv::sum(D_1_C_2)(0);

	cv::Mat D_3_C_2;
	vecMul(D, C, 2, 1, D_3_C_2);
	float sum_D_3_C_2 = cv::sum(D_3_C_2)(0);

	cv::Mat D_1_C_3;
	vecMul(D, C, 0, 2, D_1_C_3);
	float sum_D_1_C_3 = cv::sum(D_1_C_3)(0);

	cv::Mat D_2_C_1;
	vecMul(D, C, 1, 0, D_2_C_1);
	float sum_D_2_C_1 = cv::sum(D_2_C_1)(0);

	cv::Mat D_2_C_3;
	vecMul(D, C, 1, 2, D_2_C_3);
	float sum_D_2_C_3 = cv::sum(D_2_C_3)(0);

	cv::Mat D_3_C_1;
	vecMul(D, C, 2, 0, D_3_C_1);
	float sum_D_3_C_1 = cv::sum(D_3_C_1)(0);

	b = cv::Mat(6, 1, CV_32F);

	b.ptr<float>(0)[0] = sum_D_1_C_4 - sum_C_1_C_4;
	b.ptr<float>(1)[0] = sum_D_2_C_4 - sum_C_2_C_4;
	b.ptr<float>(2)[0] = sum_D_3_C_4 - sum_C_3_C_4;
	b.ptr<float>(3)[0] = sum_D_1_C_2 - sum_D_2_C_1;
	b.ptr<float>(4)[0] = sum_D_2_C_3 - sum_D_3_C_2;
	b.ptr<float>(5)[0] = sum_D_3_C_1 - sum_D_1_C_3;
}

cv::Mat point_to_point_optimize(const cv::Mat& C, const cv::Mat& D){
	
	cv::Mat A, b, x;

	point_to_point_linear(C, D, A, b);

	cv::solve(A, b, x, cv::DECOMP_CHOLESKY);
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

	return transformDelta;
}