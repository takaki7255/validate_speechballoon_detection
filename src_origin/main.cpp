//
//  main.cpp
//  main
//
//  Created by 田中海斗 on 2022/03/22.
//
#include <stdio.h>
#include <iostream>
#include <opencv2/opencv.hpp>
#include <vector>
//headerfile
#include "read_file_path.hpp"
#include "twopage_to_onepage.hpp"
#include "page_classification.hpp"
#include "frame_separation.hpp"
#include "speechballoon_separation.hpp"
#include "blackpage_framedetect.hpp"
#include "page_removing_frame.hpp"
//inputimage foldername
//#define FOLDERNAME "../img/Belmondo"
//#define SELECTED_PANELS_DIR "../result/speechballoon/Belmondo_l_先行研究/"

int main(int argc, const char * argv[]) {
    // Accept command line arguments for input/output paths
    std::string FOLDERNAME = (argc > 1) ? argv[1] : "evaluation_dataset/sampled_images";
    std::string SELECTED_PANELS_DIR = (argc > 2) ? std::string(argv[2]) + "/" : "origin_outputs/";
    
    // Create output directory
    std::string mkdir_cmd = "mkdir -p " + SELECTED_PANELS_DIR;
    system(mkdir_cmd.c_str());
    
    std::cout << "Input folder: " << FOLDERNAME << std::endl;
    std::cout << "Output folder: " << SELECTED_PANELS_DIR << std::endl;
    //Load images in a folder
    std::vector<std::string> image_paths;
    // Try both jpg and png extensions
    image_paths = ReadFilePath::get_file_path(FOLDERNAME,"jpg");
    if (image_paths.empty()) {
        image_paths = ReadFilePath::get_file_path(FOLDERNAME,"png");
    }
    
    std::cout << "Found " << image_paths.size() << " images" << std::endl;
//    for (int i=0; i<image_paths.size(); i++) {
//        std::cout<<image_paths[i]<<std::endl;
//    }

    //Variable Definition
    std::vector<cv::Mat> page_img;//A group of page images(spread cut out of the image)
    std::vector<cv::Mat> page_two_img;//two group of page images(spread cut out of the image)
    //--page classification
    std::vector<bool> page_type(2,0);//0-white 1-black
    //--frame extraction
    Framedetect panel;//instantiate
    std::vector<cv::Mat> src_panel_img;//One-page frame image
    std::vector<cv::Mat> src_panel_imgs;//All-page frame image
    //--speechballoon extraction
    Speechballoon speechballoon;//instantiate
    std::vector<cv::Mat> speechballoon_img;//One-frame speechballoon image
    std::vector<cv::Mat> speechballoon_imgs;//All-frame speechballoon image
    static int speechballoon_max = 99;//Specify maximum number of pieces
    
    //Start processing
    for (int i=0; i<image_paths.size(); i++) {
        cv::Mat img = cv::imread(image_paths[i],0);//file loading
        std::cout<<image_paths[i]<<std::endl;
        if(img.empty()) continue;
        page_img = PageCut::pageCut(img);//two pages apart
        for (int j=0; j<page_img.size(); j++) {
            page_two_img.push_back(page_img[j]);
        }
        //---start Processing on page---//
        
        //*page calassification
        for (int j=0; j<page_two_img.size(); j++) {
            page_type[j] = ClassificationPage::get_page_type(page_two_img[j]);//0-white 1-black
        }
        
        //*frame extraction
        for (int j=0; j<page_two_img.size(); j++) {
            if(page_type[j]==1) continue;//skip blackpage
            src_panel_img = panel.frame_detect(page_two_img[j]);
        }
        
        for (int j=0; j<src_panel_img.size(); j++) {
            if(page_type[j]==1) continue;//skip blackpage
            src_panel_imgs.push_back(src_panel_img[j]);
        }
        
        //---end Processing on page---//

        //---start Processing on panel---//
        for (int j=0; j<src_panel_img.size(); j++) {

            //*speechballoon extraction
            speechballoon_img = speechballoon.speechballoon_detect(src_panel_img[j]);
            
            if (speechballoon_img.empty()) continue;
            if (speechballoon_img.size() > speechballoon_max) continue;
            
            for (int index=0; index<speechballoon_img.size(); index++) {
                speechballoon_imgs.push_back(speechballoon_img[index]);
//                cv::imshow("balloon", speechballoon_img[index]);
//                cv::waitKey();
                std::ostringstream oss;
                oss << std::setfill('0') << std::setw(3) << i << '_' << j << '_' << index << ".png";
                cv::imwrite(SELECTED_PANELS_DIR + oss.str(), speechballoon_img[index] );
            }
            
        }
        //---end Processing on panel---//
    }
    //end processing

    //Image display(confirmation)
//    for (int j=0; j<src_panel_imgs.size(); j++) {
//        cv::imshow("panel", src_panel_imgs[j]);
//        cv::waitKey();
//    }
    return 0;
}
