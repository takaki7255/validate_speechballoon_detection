//
//  twopage_to_onepage.hpp
//  main
//
//  Created by 田中海斗 on 2022/05/11.
//

#ifndef twopage_to_onepage_hpp
#define twopage_to_onepage_hpp

#include <stdio.h>
#include <iostream>
#include <opencv2/opencv.hpp>
#include <vector>

class PageCut
{
public:
    static std::vector<cv::Mat> pageCut(cv::Mat &input_page_image);
private:
};
#endif /* twopage_to_onepage_hpp */
