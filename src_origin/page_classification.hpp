//
//  page_classification.hpp
//  main
//
//  Created by 田中海斗 on 2022/06/13.
//

#ifndef page_classification_hpp
#define page_classification_hpp

#include <vector>
#include <stdio.h>
#include "opencv2/opencv.hpp"
class ClassificationPage {
public:
    static bool get_page_type(cv::Mat &src_image);
    
private:
    static cv::Mat findFrameArea(cv::Mat &input_page_image);
};
#endif /* page_classification_hpp */
