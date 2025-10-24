//
//  frame_separation.hpp
//  main
//
//  Created by 田中海斗 on 2022/06/01.
//

#ifndef frame_separation_hpp
#define frame_separation_hpp

#include <vector>
#include <stdio.h>
#include "opencv2/opencv.hpp"

//直線の情報をまとめるクラス
class Line {
    //y軸をx軸とするかどうか
    bool y2x;
public:
    cv::Point p1, p2;
    
    //傾きと切片
    float a, b;
    
    //コンストラクタ
    Line(cv::Point p1, cv::Point p2, bool y2x);
    
    Line();
    
    //公式算出
    void calc();
    
    //領域の判定
    int judgeArea(cv::Point p);
};

// 四角形の4点の情報をまとめるクラス
class Points {
public:
    // 四角形の4点座標
    cv::Point lt, lb, rt, rb;
    // 四角形の上下左右線
    Line top_line, bottom_line, left_line, right_line;
    
    // コンストラクタ
    Points(cv::Point lt = { 0, 0 }, cv::Point lb = { 0, 0 }, cv::Point rt = { 0, 0 }, cv::Point rb = { 0, 0 });
    
    //直線情報の更新
    void renew_line();
    
    // 四角形の外側(直線上は含まない)にあるかの判定
    bool outside(cv::Point p);
    
};

//コマ抽出クラス　入力はページ画像
class Framedetect
{
public:
    std::vector<cv::Mat> frame_detect(cv::Mat &src_page);//コマ抽出関数
private:
    void extractSpeechBalloon(std::vector<std::vector<cv::Point> > fukidashi_contours, std::vector<cv::Vec4i> hierarchy2, cv::Mat &gaussian_img); //吹き出し抽出関数
    void findFrameExistenceArea(const cv::Mat &inverseBinImage);//コマ存在領域推定関数
    void drawHoughLines(std::vector<cv::Point2f> lines, cv::Mat &drawLinesImage);//直線描画1
    void drawHoughLines2(std::vector<cv::Point2f> lines, cv::Mat &drawLinesImage);//直線描画2
    void createAndImgWithBoundingBox(cv::Mat &src_img, const std::vector<std::vector<cv::Point>> contours,  cv::Mat inverse_bin_img, cv::Mat &dst_img);//バウンディングボックスを利用して矩形の欠損を埋める処理（階調反転二値画像と輪郭の論理積）
    bool judgeAreaOfBoundingBox(const cv::Rect bounding_box, int page_area);//バウンディングボックス面積判定関数
    void judgeBoundingBoxOverlap(bool &isOverlap,  std::vector<cv::Rect> bounding_boxes, cv::Rect brect);//バウンディングボックスが内包されてるか
    void definePanelCorners(bool &definite, cv::Point currentPoint, double &boundingBoxMinDist, cv::Point PageCornerPoint, cv::Point &definitePanelPoint, cv::Point boundingBoxPoint);//（点が決定したか，ページ角，最終的なコマ角，バウンディングボックス角）
    void align2edge(Points &definitePanelPoint,  cv::Mat inverseBinImage);//端の点を揃える
    void createAlphaImage(cv::Mat &alphaImage, Points definitePanelPoint);//コマ以外を透過する

    Points pageCorners;//角の座標を格納する変数
};

#endif /* frame_separation_hpp */
