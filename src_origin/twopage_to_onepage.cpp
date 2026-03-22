//
//  twopage_to_onepage.cpp
//  main
//
//  Created by 田中海斗 on 2022/05/11.
//

#include "twopage_to_onepage.hpp"
std::vector<cv::Mat> PageCut::pageCut(cv::Mat &input_page_image){
    std::vector<cv::Mat> page_image;
    if (input_page_image.cols > input_page_image.rows) {//縦<横の場合:見開きだと判断し真ん中で切断
        cv::Mat cut_img_left(input_page_image, cv::Rect(0, 0, input_page_image.cols/2, input_page_image.rows));//右ページ
        cv::Mat cut_img_right(input_page_image, cv::Rect(input_page_image.cols/2, 0, input_page_image.cols/2, input_page_image.rows));//左ページ
        page_image.push_back(cut_img_right);
        page_image.push_back(cut_img_left);
    }else{//縦>横の場合:単一ページ画像だと判断しそのまま保存
        page_image.push_back(input_page_image);
    }
    return page_image;
}
