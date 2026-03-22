//
//  speechballoon_separation.cpp
//  main
//
//  Created by 田中海斗 on 2022/11/29.
//

#include "speechballoon_separation.hpp"

std::vector<cv::Mat> Speechballoon::speechballoon_detect(cv::Mat &src_img){
    cv::Size imageSize(src_img.cols, src_img.rows);     //画像サイズ
    cv::Mat zeros = cv::Mat::zeros(imageSize, CV_8UC1); //初期画像情報(グレースケール)
    cv::Mat gray_img = zeros.clone();                   //グレースケール
    cv::Mat bin_img = zeros.clone();                    //二値画像
    cv::Mat sb_filled_panel_img = src_img.clone();        //パネルサイズの吹き出し塗りつぶし用
    cv::Mat alpha_img = cv::Mat::zeros(imageSize, CV_8UC1);//背景画像(透明)
    
    std::vector<cv::Mat> speechballoon_images;//return用　ページ単位の吹き出し群
    std::vector<cv::Mat> speechballoon_images_bin;//　ページ単位の吹き出し群
    
    //白黒比率判定
    cv::Mat panel_sb_blacked_img = src_img.clone(); //吹き出し黒塗り用
    cv::cvtColor(panel_sb_blacked_img, panel_sb_blacked_img, cv::COLOR_BGRA2GRAY);
    
    //グレースケールに（パネルサイズの吹き出し塗りつぶし用）
    cv::cvtColor(sb_filled_panel_img, sb_filled_panel_img, cv::COLOR_BGRA2GRAY);
    
    //二値化のためにグレースケール化
    cv::cvtColor(src_img, gray_img, cv::COLOR_BGRA2GRAY);
    
    //二値化
    cv::threshold(gray_img.clone(), bin_img, 230, 255, cv::THRESH_BINARY);
    
    //オープニング
    cv::erode(bin_img, bin_img, cv::Mat(),cv::Point(-1,-1), 1); //収縮
    cv::dilate(bin_img, bin_img, cv::Mat(),cv::Point(-1,-1), 1);  //膨張
    
    std::vector<cv::Vec4i> hierarchy;
    std::vector<std::vector<cv::Point> > contours;
    std::vector<std::vector<cv::Point> > contours_shape;
    cv::Mat canvas_img = src_img.clone();
    //輪郭抽出
    cv::findContours(bin_img, contours, hierarchy, cv::RETR_LIST, cv::CHAIN_APPROX_NONE);

    cv::Mat back_150_img(imageSize, CV_8UC1, cv::Scalar(150));  //グレー背景用
    cv::Mat mask_img(imageSize, CV_8UC1, cv::Scalar(255));  //マスク用
    
    //4チャンネルに
    cv::cvtColor(alpha_img, alpha_img, cv::COLOR_GRAY2BGRA);//GRAY->BGRA

    //全画素透明にする
    for(int i=0; i<alpha_img.rows; i++){
        for(int j=0; j<alpha_img.cols; j++){
            cv::Vec4b px = alpha_img.at<cv::Vec4b>(i,j);  //[0][1][2][3]:BGRA
            px[3]=0;
            alpha_img.at<cv::Vec4b>(i,j)=px;
        }
    }
    
    // 輪郭の描画
    for(int i=0;i<contours.size();i++){
        
        cv::Mat mask_result_1_img = src_img.clone();        //背景グレー用
        //グレースケールに（下だとエラー出るからここに置いた）
        cv::cvtColor(mask_result_1_img, mask_result_1_img, cv::COLOR_BGR2GRAY);
        
        double panel_area = src_img.rows * src_img.cols;  //コマの面積
        double area = cv::contourArea(contours[i]);  //輪郭の面積
        double length = cv::arcLength(contours[i], true);  //輪郭長
        double en = 4.0 * M_PI * area / (length * length);; //円形度
        
        
        double B=0, W=0, G=0;  //画素カウント用(B:黒, W:白, G:150)
        int TH=255/3;//85 /5
        
        //極端に小さいor大きい，または吹き出しっぽくない形の除外
        if(panel_area * 0.01 <= area && area < panel_area * 0.9 && en > 0.4){
            cv::Rect bounding_box = cv::boundingRect (contours[i]); //バウンディングボックスの取得
            cv::Size bounding_box_size(bounding_box.width,bounding_box.height);
            
            //吹き出し画像の中心座標
            int center_x, center_y=0;
            center_x = bounding_box.x+(int(bounding_box.width/2));
            center_y = bounding_box.y+(int(bounding_box.height/2));
            
            //マスク画像生成
            cv::drawContours(mask_img, contours, i, 0, 4, cv::LINE_AA, hierarchy);
            cv::drawContours(mask_img, contours, i, 0, -1, cv::LINE_AA, hierarchy);
            
            //マスク処理で背景をグレーにする(背景, 吹き出し, マスク)
            cv::copyTo(back_150_img, mask_result_1_img, mask_img);
            
            //吹き出し部分だけ切り取る
            cv::Mat resize_img(mask_result_1_img, bounding_box);
            
            //B,W,Gのカウント
            for(int y=0; y<resize_img.rows; y++){
                for(int x=0; x<resize_img.cols; x++){
                    unsigned char s = resize_img.at<unsigned char>(y,x);
                    if(s!=150){
                        if(s>(255-TH) ){  //白部分
                            W++;
                        }else if(s<TH){
                            B++;  //黒部分
                        }else{
                            G++;  //背景
                        }
                    }
                    
                }
            }
            
//            printf("---------------B: %f, W: %f, B/W: %f--------------\n", B,W, B/W);
            
            //矩形度
            double maxrect=(double)(bounding_box.width * bounding_box.height) * 0.95;
            
            //吹き出しの白黒比率判定(吹き出しらしい比率)
            if( (0.01<(B/W) && (B/W)< 0.7 )&& B>=10){ //0.0001<(b/h)
                
                //吹き出しの形判定ーーーーーーーーーーーーーーーーーーーーーーーーーー
                int setType=0;
                if(area >= maxrect){ //矩形
                    setType=1;
                    //std::cout<<"矩形"<<std::endl;
                }else if(en >= 0.7){  //円
                    setType=0;
                    //std::cout<<"丸"<<std::endl;
                }else{
                    setType=2;
                    //std::cout<<"ギザ"<<std::endl;
                }
                
                cv::Mat mask_result_2_img = src_img.clone();        //背景透明用
                
                //4チャンネルに
                cv::cvtColor(mask_result_2_img, mask_result_2_img, cv::COLOR_BGR2BGRA);
                
                //マスク処理で吹き出し以外を透明化
                cv::copyTo(alpha_img, mask_result_2_img, mask_img);
                
                if (bounding_box.width*bounding_box.height <  panel_area * 0.9) {//areaではなく切り出し後の画像サイズでコマに近しいものを除去
                    //吹き出し部分だけ切り取る
                    //保存用
                    cv::Mat resize_alpha_img(mask_result_2_img, bounding_box);
                    speechballoon_images.push_back(resize_alpha_img);
                    //二値処理用
                    cv::Mat mask_result_bin = mask_result_2_img.clone();
                    cv::threshold(mask_result_bin, mask_result_bin, 150, 255, cv::THRESH_BINARY);
                    cv::Mat resize_alpha_img_bin(mask_result_bin, bounding_box);
                    speechballoon_images_bin.push_back(resize_alpha_img_bin);
                }
            }
        }
    }
    //吹き出し誤抽出除去手法
    int balloon_px_th =5;//抽出した吹き出し画像で，画像端から何ピクセル内側までを吹き出しの画素として判断するか
    cv::Vec4b p,pt,pb,pl,pr;
    std::vector<cv::Mat> unspeechballoon_images;
    int unspeechballoon_flag;
    int black_count=0;//吹き出し内部に黒色画素が全くない場合を除去する
    int white_count=0;
    int index=0;
    std::vector<bool> index_flag;
    
    for (int j=0; j<speechballoon_images_bin.size();j++) {
        if (speechballoon_images_bin[j].empty())continue;
        unspeechballoon_flag = 0;
        black_count=0;
        
        for (int y=0; y<speechballoon_images_bin[j].rows; y++) {
            for (int x=0; x<speechballoon_images_bin[j].cols; x++) {
                cv::Vec4b p = speechballoon_images_bin[j].at<cv::Vec4b>(cv::Point(x, y));
                //p[0]=255で許容範囲をマーク
                if (x<=balloon_px_th||y<=0+balloon_px_th||x>=speechballoon_images_bin[j].cols-balloon_px_th||y>=speechballoon_images_bin[j].rows-balloon_px_th) {
                    if (p[3]!=0){
                        p[0]=255;
                        p[1]=0;
                        p[2]=0;
                    }
                }else{
                    for(int th=1;th<=balloon_px_th;th++){
                        cv::Vec4b pt = speechballoon_images_bin[j].at<cv::Vec4b>(cv::Point(x, y-th));
                        cv::Vec4b pb = speechballoon_images_bin[j].at<cv::Vec4b>(cv::Point(x, y+th));
                        cv::Vec4b pr = speechballoon_images_bin[j].at<cv::Vec4b>(cv::Point(x+th, y));
                        cv::Vec4b pl = speechballoon_images_bin[j].at<cv::Vec4b>(cv::Point(x-th, y));
                        //naname
                        cv::Vec4b ptr = speechballoon_images_bin[j].at<cv::Vec4b>(cv::Point(x+th, y-th));
                        cv::Vec4b ptl = speechballoon_images_bin[j].at<cv::Vec4b>(cv::Point(x-th, y-th));
                        cv::Vec4b pbr = speechballoon_images_bin[j].at<cv::Vec4b>(cv::Point(x+th, y+th));
                        cv::Vec4b pbl = speechballoon_images_bin[j].at<cv::Vec4b>(cv::Point(x-th, y+th));
                        if(pt[3]==0||pb[3]==0||pl[3]==0||pr[3]==0||ptr[3]==0||ptl[3]==0||pbr[3]==0||pbl[3]==0){
                            if (p[3]!=0){
                                p[0]=255;
                                p[1]=0;
                                p[2]=0;
                            }
                        }
                    }
                }
                speechballoon_images_bin[j].at<cv::Vec4b>(cv::Point(x, y)) = p;
            }
        }
        for (int y=0; y<speechballoon_images_bin[j].rows; y++) {
            for (int x=0; x<speechballoon_images_bin[j].cols; x++) {
                cv::Vec4b p = speechballoon_images_bin[j].at<cv::Vec4b>(cv::Point(x, y));

                //p[0]=255と接している黒色画素を走査

                //隣接画素
                cv::Vec4b pt = speechballoon_images_bin[j].at<cv::Vec4b>(cv::Point(x, y-1));
                cv::Vec4b pb = speechballoon_images_bin[j].at<cv::Vec4b>(cv::Point(x, y+1));
                cv::Vec4b pr = speechballoon_images_bin[j].at<cv::Vec4b>(cv::Point(x+1, y));
                cv::Vec4b pl = speechballoon_images_bin[j].at<cv::Vec4b>(cv::Point(x-1, y));
                if (p[3]!=0&&(x!=0||y!=0||x!=speechballoon_images_bin[j].cols-1||y!=speechballoon_images_bin[j].rows-1)&&
                    ((pt[0]==255&&pt[1]==0&&pt[2]==0)||(pb[0]==255&&pb[1]==0&&pb[2]==0)||(pl[0]==255&&pl[1]==0&&pl[2]==0)||(pr[0]==255&&pr[1]==0&&pr[2]==0))&&!(p[0]==255&&p[1]==0&&p[2]==0)) {//マークした許容範囲より内側を操作
                    if(p[0]==0&&p[1]==0&&p[2]==0){//黒色画素を走査
                        p[0]=0;
                        p[1]=0;
                        p[2]=255;
                    }
                }
                speechballoon_images_bin[j].at<cv::Vec4b>(cv::Point(x, y)) = p;
            }
        }
        for (int y=0; y<speechballoon_images_bin[j].rows; y++) {
            for (int x=0; x<speechballoon_images_bin[j].cols; x++) {
                cv::Vec4b p = speechballoon_images_bin[j].at<cv::Vec4b>(cv::Point(x, y));
                if (p[3]!=0&&(x!=0||y!=0||x!=speechballoon_images_bin[j].cols-1||y!=speechballoon_images_bin[j].rows-1)&&p[0]==0&&p[1]==0&&p[2]==255) {
                    unspeechballoon_flag++;
                }else if (p[3]!=0&&(p[0]==0&&p[1]==0&&p[2]==0)&&!(p[0]==0&&p[1]==0&&p[2]==255)){
                    black_count++;
                }else{
                    white_count++;
                }
            }
        }
//        std::cout<<"black:"<<black_count<<std::endl;
//        std::cout<<"white:"<<white_count<<std::endl;
        if (unspeechballoon_flag>0||black_count<100) {
            unspeechballoon_images.push_back(speechballoon_images_bin[j]);
            index_flag.push_back(false);
        }else{
            index_flag.push_back(true);
        }
        index++;
    }
//    for (int i=0;i<unspeechballoon_images.size();i++) {
//        cv::imshow("not speech balloon", unspeechballoon_images[i]);
//        std::cout<<index_flag[i]<<std::endl;
//        cv::waitKey();
//    }
    for (int i=index_flag.size()-1; i>=0; i--) {
        if (index_flag[i]==false) {
            //speechballoon_images.erase(speechballoon_images.begin()+i);
        }
    }
    return speechballoon_images;
}

// New function that returns balloons with bounding box information
std::vector<Balloon> Speechballoon::speechballoon_detect_with_bbox(cv::Mat &src_img){
    cv::Size imageSize(src_img.cols, src_img.rows);
    cv::Mat zeros = cv::Mat::zeros(imageSize, CV_8UC1);
    cv::Mat gray_img = zeros.clone();
    cv::Mat bin_img = zeros.clone();
    cv::Mat sb_filled_panel_img = src_img.clone();
    cv::Mat alpha_img = cv::Mat::zeros(imageSize, CV_8UC1);
    
    std::vector<Balloon> balloons_with_bbox;  // Return value with bbox info
    std::vector<cv::Mat> speechballoon_images;
    std::vector<cv::Rect> bounding_boxes;  // Store bboxes
    std::vector<cv::Mat> speechballoon_images_bin;
    
    // Same processing as original
    cv::Mat panel_sb_blacked_img = src_img.clone();
    cv::cvtColor(panel_sb_blacked_img, panel_sb_blacked_img, cv::COLOR_BGRA2GRAY);
    cv::cvtColor(sb_filled_panel_img, sb_filled_panel_img, cv::COLOR_BGRA2GRAY);
    cv::cvtColor(src_img, gray_img, cv::COLOR_BGRA2GRAY);
    cv::threshold(gray_img.clone(), bin_img, 230, 255, cv::THRESH_BINARY);
    cv::erode(bin_img, bin_img, cv::Mat(),cv::Point(-1,-1), 1);
    cv::dilate(bin_img, bin_img, cv::Mat(),cv::Point(-1,-1), 1);
    
    std::vector<cv::Vec4i> hierarchy;
    std::vector<std::vector<cv::Point>> contours;
    cv::Mat canvas_img = src_img.clone();
    cv::findContours(bin_img, contours, hierarchy, cv::RETR_LIST, cv::CHAIN_APPROX_NONE);

    cv::Mat back_150_img(imageSize, CV_8UC1, cv::Scalar(150));
    cv::Mat mask_img(imageSize, CV_8UC1, cv::Scalar(255));
    cv::cvtColor(alpha_img, alpha_img, cv::COLOR_GRAY2BGRA);

    for(int i=0; i<alpha_img.rows; i++){
        for(int j=0; j<alpha_img.cols; j++){
            cv::Vec4b px = alpha_img.at<cv::Vec4b>(i,j);
            px[3]=0;
            alpha_img.at<cv::Vec4b>(i,j)=px;
        }
    }
    
    for(int i=0; i<contours.size(); i++){
        cv::Mat mask_result_1_img = src_img.clone();
        cv::cvtColor(mask_result_1_img, mask_result_1_img, cv::COLOR_BGR2GRAY);
        
        double panel_area = src_img.rows * src_img.cols;
        double area = cv::contourArea(contours[i]);
        double length = cv::arcLength(contours[i], true);
        double en = 4.0 * M_PI * area / (length * length);
        
        double B=0, W=0, G=0;
        int TH=255/3;
        
        if(panel_area * 0.01 <= area && area < panel_area * 0.9 && en > 0.4){
            cv::Rect bounding_box = cv::boundingRect(contours[i]);
            cv::Size bounding_box_size(bounding_box.width, bounding_box.height);
            
            int center_x = bounding_box.x+(int(bounding_box.width/2));
            int center_y = bounding_box.y+(int(bounding_box.height/2));
            
            cv::drawContours(mask_img, contours, i, 0, 4, cv::LINE_AA, hierarchy);
            cv::drawContours(mask_img, contours, i, 0, -1, cv::LINE_AA, hierarchy);
            cv::copyTo(back_150_img, mask_result_1_img, mask_img);
            cv::Mat resize_img(mask_result_1_img, bounding_box);
            
            for(int y=0; y<resize_img.rows; y++){
                for(int x=0; x<resize_img.cols; x++){
                    unsigned char s = resize_img.at<unsigned char>(y,x);
                    if(s!=150){
                        if(s>(255-TH)){
                            W++;
                        }else if(s<TH){
                            B++;
                        }else{
                            G++;
                        }
                    }
                }
            }
            
            double maxrect=(double)(bounding_box.width * bounding_box.height) * 0.95;
            
            if((0.01<(B/W) && (B/W)< 0.7) && B>=10){
                int setType=0;
                if(area >= maxrect){
                    setType=1;
                }else if(en >= 0.7){
                    setType=0;
                }else{
                    setType=2;
                }
                
                cv::Mat canvas_sb = alpha_img.clone();
                cv::drawContours(canvas_sb, contours, i, cv::Scalar(255,255,255,255), -1, cv::LINE_AA, hierarchy);
                
                for(int y=bounding_box.y; y<bounding_box.y+bounding_box.height; y++){
                    for(int x=bounding_box.x; x<bounding_box.x+bounding_box.width; x++){
                        cv::Vec4b alpha_pix = canvas_sb.at<cv::Vec4b>(y,x);
                        if(alpha_pix[3]!=0){
                            cv::Vec4b src_pix = src_img.at<cv::Vec4b>(y,x);
                            canvas_sb.at<cv::Vec4b>(y,x) = src_pix;
                        }
                    }
                }
                
                cv::Mat balloon_crop(canvas_sb, bounding_box);
                speechballoon_images.push_back(balloon_crop);
                bounding_boxes.push_back(bounding_box);  // Store bbox
            }
            
            cv::drawContours(mask_img, contours, i, 255, -1, cv::LINE_AA, hierarchy);
        }
    }
    
    // Create Balloon objects with image, bbox, and mask
    for(size_t i=0; i<speechballoon_images.size(); i++){
        Balloon balloon;
        balloon.img = speechballoon_images[i];
        balloon.bbox = bounding_boxes[i];
        
        // Extract alpha channel from balloon image and create binary mask
        if(speechballoon_images[i].channels() == 4) {
            std::vector<cv::Mat> channels;
            cv::split(speechballoon_images[i], channels);
            cv::threshold(channels[3], balloon.mask, 0, 255, cv::THRESH_BINARY);
        } else {
            // Fallback: create mask from grayscale
            cv::Mat gray;
            if(speechballoon_images[i].channels() == 3) {
                cv::cvtColor(speechballoon_images[i], gray, cv::COLOR_BGR2GRAY);
            } else {
                gray = speechballoon_images[i].clone();
            }
            cv::threshold(gray, balloon.mask, 250, 255, cv::THRESH_BINARY);
        }
        
        balloons_with_bbox.push_back(balloon);
    }
    
    return balloons_with_bbox;
}

