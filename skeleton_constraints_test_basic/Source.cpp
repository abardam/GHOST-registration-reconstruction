#include <recons_optimization.h>
#include <opencv2\opencv.hpp>

int main(){

	float prev_f[] = { 0.9998197, -0.00075546803, 0.019461991, 0.00011396408,
	0.017121172, 0.543531, -0.90810275, 417.5798,
	-0.011044296, 0.90825808, 0.54342592, -0.00012207031,
	0, 0, 0, 1 };

	float ref_f[] = { 0.99980766, 1.3373792e-006, 0.019611973, 0.00011398199,
	0.017048875, 0.49420887, -0.86917603, 417.5798,
	-0.0096935723, 0.86934316, 0.4941138, -0.000129293,
	0, 0, 0, 1 };

	cv::Mat prev(4, 4, CV_32F, prev_f);
	cv::Mat ref(4, 4, CV_32F, ref_f);

	//cv::Mat prev = cv::Mat::eye(4, 4, CV_32F);
	//cv::Mat ref = cv::Mat::eye(4, 4, CV_32F);
	//
	//prev.ptr<float>(0)[3] = -50;
	//prev.ptr<float>(1)[3] = -5;
	//prev.ptr<float>(2)[3] = -50;
	//cv::Vec3f rot1(0, 0.01, 0);
	//cv::Rodrigues(rot1, prev(cv::Range(0, 3), cv::Range(0, 3)));
	//
	//cv::Mat offset = cv::Mat::eye(4, 4, CV_32F);
	//offset.ptr<float>(0)[3] = 50000;
	//offset.ptr<float>(1)[3] = 5000;
	//offset.ptr<float>(2)[3] = 50000;
	//
	//cv::Vec3f rot(0.01,0,0);
	//cv::Rodrigues(rot, ref(cv::Range(0, 3), cv::Range(0, 3)));
	//ref.ptr<float>(0)[3] = 0;
	//ref.ptr<float>(1)[3] = 0;
	//ref.ptr<float>(2)[3] = 0;
	//
	//ref = offset.inv() * ref * offset;

	while (true){
		cv::Mat delta = skeleton_constraints_optimize(prev, ref, 1, 1);

		std::cout << "delta * prev" << std::endl << delta * prev << std::endl <<
			"ref" << std::endl << ref << std::endl <<
			"diff" << std::endl << delta * prev - ref << std::endl;

		char a;
		std::cin >> a;

		if (a == 'q') break;

		prev = delta * prev;
	}
}