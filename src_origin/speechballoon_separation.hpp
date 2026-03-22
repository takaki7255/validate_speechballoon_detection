//
//  speechballoon_separation.hpp
//  main
//
//  Created by 田中海斗 on 2022/11/29.
//

#ifndef speechballoon_separation_hpp
#define speechballoon_separation_hpp

#include <vector>
#include <stdio.h>
#include "opencv2/opencv.hpp"

// Structure to hold balloon image, bounding box, and mask
struct Balloon {
    cv::Mat img;     // Balloon image (BGRA)
    cv::Rect bbox;   // Position in panel coordinates
    cv::Mat mask;    // Binary mask (bbox size, CV_8UC1, 0/255)
};

class Speechballoon
{
    public:
        std::vector<cv::Mat> speechballoon_detect(cv::Mat &src_img);//コマ画像 (original)
        std::vector<Balloon> speechballoon_detect_with_bbox(cv::Mat &src_img);//コマ画像 (with position info)
    
    private:
};
#endif /* speechballoon_separation_hpp */
