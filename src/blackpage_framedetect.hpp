//
//  blackpage_framedetect.hpp
//  main
//
//  Created by 田中海斗 on 2022/03/22.
//

#ifndef blackpage_framedetect_hpp
#define blackpage_framedetect_hpp

#include <stdio.h>
#include <iostream>
#include <opencv2/opencv.hpp>

class BFramedetect
{
public:
    void blackpageFramedetect(cv::Mat &input_page_image);//黒ページのコマを抽出する関数
    static cv::Mat findFrameArea(cv::Mat &input_page_image);//黒ページ存在領域推定関数(余白除去)
    void createHist(std::vector<int> histgram_num);//ヒストグラムを生成する関数
private:
    
};
#endif /* blackpage_framedetect_hpp */
