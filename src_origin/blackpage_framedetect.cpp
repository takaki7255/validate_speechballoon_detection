//
//  blackpage_framedetect.cpp
//  main
//
//  Created by 田中海斗 on 2022/03/22.
//

#include "blackpage_framedetect.hpp"

#define COLOR_MAX 255//最大輝度値

//未完成の関数です(課題：黒色のページからコマを抽出する)

//過去回想などで用いられる黒のページからコマを抽出する関数
void BFramedetect::blackpageFramedetect(cv::Mat &input_page_image)
{
    //黒画素のみを領域検出し，小さすぎるものを省く
    cv::Mat black_only_image = input_page_image;//黒画素以外塗りつぶされた画像
    for(int y=0; y<input_page_image.rows; y++){
        for(int x=0; x<input_page_image.cols; x++){
            if(input_page_image.at<uchar>(y,x) < 10){
                black_only_image.at<uchar>(y,x)=0;
            }else{
                black_only_image.at<uchar>(y,x)=255;
            }
        }
    }
    std::vector< std::vector<cv::Point> > contours;
    std::vector<cv::Vec4i> hierarchy;
    cv::findContours(black_only_image, contours, hierarchy, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);
    cv::cvtColor(black_only_image, black_only_image, cv::COLOR_GRAY2BGR);
    
    cv::drawContours(black_only_image,contours,-1,cv::Scalar(0,0,255),-1);
//    cv::imshow("black_only", black_only_image);
//    cv::waitKey();
    //画像横方向で黒画素の走査
    std::vector<int> histgram_src_lr(input_page_image.cols, 0);

    for(int y=0; y<input_page_image.rows; y++){
        for(int x=0; x<input_page_image.cols; x++){
            if(input_page_image.at<uchar>(y,x) == 0){
                histgram_src_lr[x]++;
            }
        }
    }
    //ヒストグラム表示(確認用)
    createHist(histgram_src_lr);
    //度数ごとに存在しているフレームを監視
    int black_num_max = 0;//ヒストグラムの度数最大値記録用
    for (int i=0; i<histgram_src_lr.size(); i++) {
        if (black_num_max < histgram_src_lr[i]) {
            black_num_max = histgram_src_lr[i];//更新
        }
    }
    std::vector<int> slices_lr(black_num_max, 0);//度数ごとの対応フレーム数を記録
    for (int y=0; y<black_num_max; y++) {
        for (int x=0; x<input_page_image.cols; x++) {
            if (histgram_src_lr[x] > y) {
                slices_lr[y]++;
            }
        }
    }
    //createHist(slices_lr);
    //画像横方向で黒画素の走査
    std::vector<int> histgram_src_tb(input_page_image.rows, 0);
    
    for(int y=0; y<input_page_image.rows; y++){
        for(int x=0; x<input_page_image.cols; x++){
            if(input_page_image.at<uchar>(y,x) == 0){
                histgram_src_tb[y]++;
            }
        }
    }
    //ヒストグラム表示(確認用)
    //createHist(histgram_src_tb);
}
//ヒストグラム（白色画素の度数分布）によるコマ存在領域の決定する関数　ページに存在する余分な部分を取り除く
cv::Mat BFramedetect::findFrameArea(cv::Mat &input_page_image)
{
    //ガウシアンフィルタ
    cv::Mat gaussian_img;
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
//ヒストグラムを生成する関数
void BFramedetect::createHist(std::vector<int> histgram_num){
    //最大値の記録(正規化用)
    int black_num_max = 0;
    for (int i=0; i<histgram_num.size(); i++) {
        if (black_num_max < histgram_num[i]) {
            black_num_max = histgram_num[i];//更新
        }
    }
    //ヒストグラム生成用の画像を作成
    cv::Mat image_hist = cv::Mat::zeros(cv::Size(355,340), CV_8UC3);//グラフサイズは255*300を想定
    //背景を描画(見やすくするためにヒストグラム部分の背景をグレーにする)
    rectangle(image_hist, cv::Point(50, 10),cv::Point(304, 320), cv::Scalar(230, 230, 230), -1);
    cv::line(image_hist, cv::Point(50,170), cv::Point(304,170), cv::Scalar(140,140,140), 1, 8, 0);//真ん中の線
    //結果を描画
    for (int i=0; i<histgram_num.size(); i++) {
        int histgram_nomalization_x=(255*i)/histgram_num.size();//x軸正規化
        int histgram_nomalization_y=(300*histgram_num[i])/black_num_max;//y軸正規化
        
        cv::line(image_hist, cv::Point(50+histgram_nomalization_x,320), cv::Point(50+histgram_nomalization_x,320-histgram_nomalization_y), cv::Scalar(255,0,0), 1, 8, 0);
        
    }
    cv::String memory_center_y,memory_top_y;//グラフの値描画用
    memory_center_y = std::to_string(black_num_max/2);
    memory_top_y = std::to_string(black_num_max);
    cv::Point text_center_pos(5,170);// center the text
    cv::Point text_top_pos(5,20);// top the text
    cv::Point text_zero_pos(5,320);// top the text
    putText(image_hist, memory_center_y, text_center_pos, cv::FONT_HERSHEY_SCRIPT_SIMPLEX, 0.5, cv::Scalar::all(255), 1.5, 8);
    putText(image_hist, memory_top_y, text_top_pos, cv::FONT_HERSHEY_SCRIPT_SIMPLEX, 0.5, cv::Scalar::all(255), 1.5, 8);
    putText(image_hist, "0", text_zero_pos, cv::FONT_HERSHEY_SCRIPT_SIMPLEX, 0.5, cv::Scalar::all(255), 1.5, 8);
//    cv::imshow("hist", image_hist);
//    cv::waitKey();
}
