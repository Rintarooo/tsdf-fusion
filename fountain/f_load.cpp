// #include "f_load.h"
#include <iostream>
#include <string>    // string
#include <fstream>   // ifstream, ofstream
#include <sstream>   // stringstream
#include <vector>
#include <opencv2/opencv.hpp>
#include <iomanip> // https://stackoverflow.com/questions/225362/convert-a-number-to-a-string-with-specified-length-in-c


void file_writer(const std::string &filename,
    const cv::Mat &Rcw, const cv::Mat &tcw){
    std::ofstream file2;
    file2.open(filename, std::ios::out);
    float* Rcw_ptr = (float*)Rcw.data;// Mat.data returns uchar*, so cast into float* 
    float R00, R01, R02;
    float R10, R11, R12;
    float R20, R21, R22;
   
    R00 = Rcw_ptr[0];
    R01 = Rcw_ptr[1];
    R02 = Rcw_ptr[2];
    R10 = Rcw_ptr[3];
    R11 = Rcw_ptr[4];
    R12 = Rcw_ptr[5];
    R20 = Rcw_ptr[6];
    R21 = Rcw_ptr[7];
    R22 = Rcw_ptr[8];
    
    float* tcw_ptr = (float*)tcw.data;// Mat.data returns uchar*, so cast into float*
    float tx, ty, tz;
    tx = tcw_ptr[0];
    ty = tcw_ptr[1];
    tz = tcw_ptr[2];

    file2 << R00 << R01 << R02 << std::endl;
    file2 << R10 << R11 << R12 << std::endl;
    file2 << R20 << R21 << R22 << std::endl;
    file2 << tx << ty << tz << std::endl;
    file2.close();  
}

void getRcw_fromRwc(const cv::Mat &Rwc, const cv::Mat &twc,
    cv::Mat &Rcw, cv::Mat &tcw)
{
    Rcw =  Rwc.t();// Rcw = Rwc^-1, Rwc is rotation matrix from camera frame to world frame.
    tcw = -Rwc.t()*twc;// slide 28/80. https://www.slideshare.net/SSII_Slides/ssii2019ts3-149136612/28
}

void file_loader (const std::string &filename, 
	const std::string &filename_dst)
{
	// https://qiita.com/Reed_X1319RAY/items/098596cda78e9c1a6bad
	std::ifstream ifs(filename, std::ios::in);
	if(!ifs.is_open()){
		std::cerr << "Error, cannot open file, check argv: " << filename << std::endl;
		std::exit(1); 
	}
   std::string line;
   // // skip 2 line
   // for(int i = 0; i < 2; i++){
   // 	std::getline(ifs, line);
   // }

   // line1
   std::getline(ifs, line);
   std::stringstream ss0(line);// ss << line;
   float R00, R01, R02;
   ss0 >> R00 >> R01 >> R02; 

   // line2
   std::getline(ifs, line);
   std::stringstream ss1(line);// ss << line;
   float R10, R11, R12;
   ss1 >> R10 >> R11 >> R12; 
   
    // line3
   std::getline(ifs, line);
   std::stringstream ss2(line);// ss << line;
   float R20, R21, R22;
   ss2 >> R20 >> R21 >> R22; 
  
    // line4
   std::getline(ifs, line);
   std::stringstream ss3(line);// ss << line;
   float tx, ty, tz;
   ss3 >> tx >> ty >> tz; 
  
   cv::Mat Rwc = (cv::Mat_<float>(3,3) <<
				   R00, R01, R02,
				   R10, R11, R12,
				   R20, R21, R22);
	cv::Mat twc = (cv::Mat_<float>(3,1) <<
				   tx, ty, tz);

	cv::Mat Rcw, tcw;
	getRcw_fromRwc(Rwc, twc, Rcw, tcw);
	
	file_writer(filename_dst, Rcw, tcw);
    ifs.close();
}

int main(int argc, char* argv[]){
	// if (argc < 2){
 //        std::cout << "Usage: ./build/main src dst\n";
 //        std::cerr << "argc: " << argc << "should be 3\n";
 //        return 1;
 //    }
 //    const std::string src= argv[1];;
 //    const std::string dst= argv[2];;
    
    // const std::string src= "inv2-pose.txt";
    // const std::string dst= "inv2-pose-new.txt";
	for (int i = 2; i <= 8; i++){
        const std::string src= "inv" + std::to_string(i) + "-pose.txt";
        const std::string dst= "inv" + std::to_string(i) + "-pose-new.txt";
        file_loader(src, dst);
    }
    return 0;
}