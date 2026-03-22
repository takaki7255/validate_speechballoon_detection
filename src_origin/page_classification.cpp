//
//  page_classification.cpp
//  main
//
//  Created by 田中海斗 on 2022/06/13.
//

#include "page_classification.hpp"
#define BLACK_LENGTH_TH 5
bool ClassificationPage::get_page_type(cv::Mat &src_image){
    cv::Mat input_page_image = src_image.clone();
    bool page_type;//0-white 1-black
    //determine the area where frames exist
    cv::Mat frame_exist_page = findFrameArea(input_page_image);
    cv::threshold(frame_exist_page, frame_exist_page, 0, 255, cv::THRESH_BINARY|cv::THRESH_OTSU);
    
    //top
    page_type = true;
    for (int y=3; y<BLACK_LENGTH_TH; y++) {
        for (int x=0; x<frame_exist_page.cols; x++) {
            if (frame_exist_page.at<uchar>(y,x)!=0) {
                page_type = false;
                break;
            }
        }
        if (page_type == false) {
            break;
        }
    }
    if (page_type == true) {
//        cv::imshow("page frame exist", frame_exist_page);
//        std::cout <<"top"<<std::endl;
//        cv::waitKey();
        return page_type;
    }
    //bottom
    page_type = true;
    for (int y=frame_exist_page.rows-3; y>frame_exist_page.rows - BLACK_LENGTH_TH; y--) {
        for (int x=0; x<frame_exist_page.cols; x++) {
            if (frame_exist_page.at<uchar>(y,x)!=0) {
                page_type = false;
                break;
            }
        }
        if (page_type == false) {
            break;
        }
    }
    if (page_type == true) {
//        cv::imshow("page frame exist", frame_exist_page);
//        std::cout <<"bottom"<<std::endl;
//        cv::waitKey();
        return page_type;
    }
    //right
    page_type = true;
    for (int y=0; y<frame_exist_page.rows; y++) {
        for (int x=frame_exist_page.cols-3; x>frame_exist_page.cols-BLACK_LENGTH_TH; x--) {
            if (frame_exist_page.at<uchar>(y,x)!=0) {
                page_type = false;
                break;
            }
        }
        if (page_type == false) {
            break;
        }
    }
    if (page_type == true) {
//        cv::imshow("page frame exist", frame_exist_page);
//        std::cout <<"right"<<std::endl;
//        cv::waitKey();
        return page_type;
    }
    //left
    page_type = true;
    for (int y=0; y<frame_exist_page.rows; y++) {
        for (int x=3; x<BLACK_LENGTH_TH; x++) {
            if (frame_exist_page.at<uchar>(y,x)!=0) {
                page_type = false;
                break;
            }
        }
        if (page_type == false) {
            break;
        }
    }
    if (page_type == true) {
//        cv::imshow("page frame exist", frame_exist_page);
//        std::cout <<"left"<<std::endl;
//        cv::waitKey();
        return page_type;
    }
    //cv::imshow("page frame exist", frame_exist_page);
    //std::cout <<"white"<<std::endl;
    //cv::waitKey();
    return page_type;
}
//determine the area where frames exist (for histgram white pixel)
cv::Mat ClassificationPage::findFrameArea(cv::Mat &input_page_image)
{
    //ガウシアンフィルタ
    cv::Mat gaussian_img = cv::Mat::zeros(input_page_image.size(), CV_8UC1);
    cv::GaussianBlur(input_page_image, gaussian_img, cv::Size(3,3), 0);  //平滑化
    //階調反転二値画像の生成
    cv::Mat inverse_bin_img = ~gaussian_img;  //階調反転画像
    cv::threshold(inverse_bin_img, inverse_bin_img, 0, 255, cv::THRESH_BINARY | cv::THRESH_OTSU);   //二値化
    
    //左右の除去
    //ヒストグラムの生成
    int *histgram_lr;//left right
    histgram_lr = (int *)calloc(inverse_bin_img.cols, sizeof(int));
    
    for(int y=0; y<inverse_bin_img.rows; y++){
        for(int x=0; x<inverse_bin_img.cols; x++){
            if(x<=2 || x>=inverse_bin_img.cols-2 || y<=2 || y>=inverse_bin_img.cols-2){
                continue;
            }
            if(0<inverse_bin_img.at<uchar>(y,x)){
                histgram_lr[x]++;
                
            }
        }
    }
    int min_x_lr=0, max_x_lr=inverse_bin_img.cols-1;
    for(int x=0; x<inverse_bin_img.cols; x++){
        if(0<histgram_lr[x]){
            min_x_lr = x;
            break;
        }
    }
    for(int x=inverse_bin_img.cols-1; x>=0; x--){
        if(0<histgram_lr[x]){
            max_x_lr = x;
            break;
        }
    }
    //誤差は両端に寄せる
    if(min_x_lr<6)min_x_lr=0;
    if(max_x_lr>inverse_bin_img.cols-6)max_x_lr=inverse_bin_img.cols;
    cv::Rect roi_lr(min_x_lr, 0, max_x_lr-min_x_lr, inverse_bin_img.rows);
    cv::Mat cut_page_img_lr(input_page_image, roi_lr);
    
    //上下の除去
    //ヒストグラムの生成
    int *histgram_tb;//top bottom
    histgram_tb = (int *)calloc(inverse_bin_img.rows, sizeof(int));
    
    for(int y=0; y<inverse_bin_img.rows; y++){
        for(int x=0; x<inverse_bin_img.cols; x++){
            if(x<=2 || x>=inverse_bin_img.rows-2 || y<=2 || y>=inverse_bin_img.rows-2){
                continue;
            }
            if(0<inverse_bin_img.at<uchar>(y,x)){
                histgram_tb[y]++;
                
            }
        }
    }
    
    int min_y_tb=0, max_y_tb=inverse_bin_img.cols-1;
    for(int y=0; y<inverse_bin_img.rows; y++){
        if(0<histgram_tb[y]){
            min_y_tb = y;
            break;
        }
    }
    for(int y=inverse_bin_img.rows-1; y>=0; y--){
        if(0<histgram_tb[y]){
            max_y_tb = y;
            break;
        }
    }
    //誤差は両端に寄せる
    if(min_y_tb<6)min_y_tb=0;
    if(max_y_tb>cut_page_img_lr.rows-6)max_y_tb=cut_page_img_lr.rows;
    //std::cerr << "min_y:" << min_y_tb <<std::endl;
    //std::cerr << "max_y:" << max_y_tb <<std::endl;
    cv::Rect roi_tb(0, min_y_tb, cut_page_img_lr.cols,max_y_tb-min_y_tb);
    cv::Mat cut_page_img(cut_page_img_lr, roi_tb);
    return cut_page_img;
}
