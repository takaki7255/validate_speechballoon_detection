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
class Speechballoon
{
    public:
        struct Balloon {
            cv::Mat img;    // cropped RGBA image of the balloon
            cv::Rect bbox;  // bbox of the crop relative to the input page
            Balloon() {}
            Balloon(const cv::Mat &i, const cv::Rect &b): img(i), bbox(b) {}
        };

        // Return vector of Balloon (cropped image + bbox in page coords)
        std::vector<Balloon> speechballoon_detect(cv::Mat &src_img);//コマ画像

    private:
};
#endif /* speechballoon_separation_hpp */
