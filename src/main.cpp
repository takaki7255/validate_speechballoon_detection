//g++ * -std=c++11 `pkg-config --cflags --libs opencv4`
//  main.cpp
//  main
//
//  Created by 田中海斗 on 2022/03/22.
//
#include <stdio.h>
#include <iostream>
#include <opencv2/opencv.hpp>
#include <vector>
#include <sstream>
//headerfile
#include "read_file_path.hpp"
#include "twopage_to_onepage.hpp"
#include "page_classification.hpp"
#include "frame_separation.hpp"
#include "speechballoon_separation.hpp"
#include "blackpage_framedetect.hpp"
#include "page_removing_frame.hpp"
// NOTE: This main was adapted to be used as an evaluation runner.
// Usage: ./speechballoon_detector <input_folder> <output_folder>

int main(int argc, const char * argv[]) {
    if (argc < 3) {
        std::cout << "Usage: " << argv[0] << " <input_folder> <output_folder>" << std::endl;
        return 1;
    }

    std::string input_folder = argv[1];
    std::string output_folder = argv[2];

    bool debug_mode = false;
    for (int ai = 1; ai < argc; ++ai) {
        std::string a = argv[ai];
        if (a == std::string("--debug")) debug_mode = true;
    }

    // create output subdirs: eval_masks and eval_instances
    std::string eval_masks_dir = output_folder + "/eval_masks/";
    std::string eval_instances_dir = output_folder + "/eval_instances/";

    // attempt to create directories (POSIX)
    std::string mkdir_cmd = "mkdir -p \"" + eval_masks_dir + "\" \"" + eval_instances_dir + "\"";
    system(mkdir_cmd.c_str());

    if (debug_mode) {
        std::string dbg = output_folder + "/debug";
        std::string mkdir_dbg = "mkdir -p \"" + dbg + "/panels\" \"" + dbg + "/balloons\" \"" + dbg + "/overlays\"";
        system(mkdir_dbg.c_str());
    }

    //Load images in a folder
    std::vector<std::string> image_paths;
    image_paths = ReadFilePath::get_file_path(input_folder,"jpg");
//    for (int i=0; i<image_paths.size(); i++) {
//        std::cout<<image_paths[i]<<std::endl;
//    }

    //Variable Definition
    std::vector<cv::Mat> page_img;//A group of page images(spread cut out of the image)
//    std::vector<cv::Mat> page_two_img;//two group of page images(spread cut out of the image)
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
        std::vector<cv::Mat> page_two_img;
//        if(i<60){
//            continue;
//        }
    cv::Mat img = cv::imread(image_paths[i],0);//file loading (grayscale)
    std::cout<<image_paths[i]<<std::endl;
    if(img.empty()) continue;
    // Use PageCut to split spreads into pages (keeps original workflow)
    page_img = PageCut::pageCut(img);//two pages apart
        for (int j=0; j<page_img.size(); j++) {
            page_two_img.push_back(page_img[j]);
        }
        //---start Processing on page---//
        
        //*page calassification
        for (int j=0; j<page_two_img.size(); j++) {
            page_type[j] = ClassificationPage::get_page_type(page_two_img[j]);//0-white 1-black
        }
        printf("pageclass");
        
        //*frame extraction
        for (int j=0; j<page_two_img.size(); j++) {
            if(page_type[j]==1) continue;//skip blackpage
            src_panel_img = panel.frame_detect(page_two_img[j]);
            for(int k=0;k<src_panel_img.size();k++){
                src_panel_imgs.push_back(src_panel_img[k]);
                // Save panel images if needed (optional)
            }
        }
//        printf("panel");
//
//        for (int j=0; j<src_panel_img.size(); j++) {
//            if(page_type[j]==1) continue;//skip blackpage
//            printf("before push panel imgs");
//            src_panel_imgs.push_back(src_panel_img[j]);
//            cv::imwrite(SELECTED_PANELS_DIR + std::to_string(i) + '_' + std::to_string(j) + ".png", src_panel_img[j] );
//        }
        
        //---end Processing on page---//

        //---Process each page as parts of a single evaluation image---//
        // We'll composite detections from all non-black pages into one pair of masks per original image.
        int orig_h = img.rows;
        int orig_w = img.cols;
        cv::Mat binary_mask = cv::Mat::zeros(orig_h, orig_w, CV_8UC1);
        cv::Mat instance_mask = cv::Mat::zeros(orig_h, orig_w, CV_8UC1);
        int global_instance_counter = 0;

        for (int j=0; j<page_two_img.size(); j++) {
            if (page_type[j] == 1) continue; // skip black pages

            // For evaluation, compose per-page detections onto the original image canvas
            cv::Mat page = page_two_img[j];
            cv::Mat page_bgr;
            if (page.channels() == 1) cv::cvtColor(page, page_bgr, cv::COLOR_GRAY2BGR);
            else page_bgr = page;

            std::vector<Speechballoon::Balloon> detected = speechballoon.speechballoon_detect(page_bgr);

            // Compute page offset: when two-page spread (PageCut returned right, left), place pages accordingly
            int page_offset_x = 0;
            if (img.cols > img.rows && page_two_img.size() == 2) {
                // PageCut returns [right, left]
                if (j == 0) page_offset_x = orig_w / 2; // right page sits on the right half
                else page_offset_x = 0; // left page sits on the left half
            }

            // Debug: save the page/panel image
            if (debug_mode) {
                std::string img_path = image_paths[i];
                size_t pos = img_path.find_last_of("/\\");
                std::string base = (pos == std::string::npos) ? img_path : img_path.substr(pos + 1);
                size_t dot = base.find_last_of('.');
                std::string name = (dot == std::string::npos) ? base : base.substr(0, dot);
                std::ostringstream panel_fn;
                panel_fn << output_folder << "/debug/panels/" << name << "_" << j << ".png";
                cv::imwrite(panel_fn.str(), page);
            }

            for (int idx = 0; idx < (int)detected.size(); ++idx) {
                cv::Mat balloon = detected[idx].img;
                if (balloon.empty()) continue;

                cv::Mat contour_mask;
                if (balloon.channels() == 4) {
                    std::vector<cv::Mat> ch;
                    cv::split(balloon, ch);
                    contour_mask = ch[3];
                } else {
                    cv::Mat gray;
                    if (balloon.channels() == 3) cv::cvtColor(balloon, gray, cv::COLOR_BGR2GRAY);
                    else gray = balloon;
                    // Create mask of non-white pixels to better follow balloon contour
                    cv::Mat nonwhite;
                    cv::threshold(gray, nonwhite, 250, 255, cv::THRESH_BINARY_INV);
                    if (cv::countNonZero(nonwhite) == 0) {
                        // fallback to Otsu if thresholding by 250 yields nothing
                        cv::threshold(gray, nonwhite, 0, 255, cv::THRESH_BINARY_INV | cv::THRESH_OTSU);
                    }
                    contour_mask = nonwhite;
                }

                std::vector<std::vector<cv::Point>> contours;
                std::vector<cv::Vec4i> hierarchy;
                cv::findContours(contour_mask, contours, hierarchy, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

                // Use bbox returned by detector to map crop coordinates back to page coords
                int local_offset_x = detected[idx].bbox.x;
                int local_offset_y = detected[idx].bbox.y;

                global_instance_counter++;
                int instance_id = std::min(global_instance_counter, 254);

                for (size_t c = 0; c < contours.size(); ++c) {
                    std::vector<cv::Point> cnt = contours[c];
                    for (auto &p : cnt) {
                        p.x += local_offset_x + page_offset_x;
                        p.y += local_offset_y;
                    }
                    cv::drawContours(binary_mask, std::vector<std::vector<cv::Point>>{cnt}, -1, cv::Scalar(255), cv::FILLED);
                    cv::drawContours(instance_mask, std::vector<std::vector<cv::Point>>{cnt}, -1, cv::Scalar(instance_id), cv::FILLED);
                }

                // Debug: save balloon crop and overlay on original image
                if (debug_mode) {
                    std::string img_path = image_paths[i];
                    size_t pos = img_path.find_last_of("/\\");
                    std::string base = (pos == std::string::npos) ? img_path : img_path.substr(pos + 1);
                    size_t dot = base.find_last_of('.');
                    std::string name = (dot == std::string::npos) ? base : base.substr(0, dot);
                    std::ostringstream bfn;
                    bfn << output_folder << "/debug/balloons/" << name << "_" << j << "_" << idx << ".png";
                    cv::imwrite(bfn.str(), balloon);

                    // overlay on original image (convert grayscale to BGR if needed)
                    cv::Mat overlay;
                    if (img.channels() == 1) cv::cvtColor(img, overlay, cv::COLOR_GRAY2BGR);
                    else overlay = img.clone();
                    for (size_t c = 0; c < contours.size(); ++c) {
                        std::vector<cv::Point> cnt2 = contours[c];
                        for (auto &p : cnt2) {
                            p.x += local_offset_x + page_offset_x;
                            p.y += local_offset_y;
                        }
                        cv::drawContours(overlay, std::vector<std::vector<cv::Point>>{cnt2}, -1, cv::Scalar(0,0,255), 2);
                    }
                    std::ostringstream ofn;
                    ofn << output_folder << "/debug/overlays/" << name << "_" << j << "_" << idx << ".png";
                    cv::imwrite(ofn.str(), overlay);
                }
            }
        }

        // Save masks with a filename derived from input image (single output per original image)
        std::string img_path = image_paths[i];
        // extract basename
        size_t pos = img_path.find_last_of("/\\");
        std::string base = (pos == std::string::npos) ? img_path : img_path.substr(pos + 1);
        // remove extension
        size_t dot = base.find_last_of('.');
        std::string name = (dot == std::string::npos) ? base : base.substr(0, dot);
        std::string outname = name;

        cv::imwrite(eval_masks_dir + outname + ".png", binary_mask);
        cv::imwrite(eval_instances_dir + outname + ".png", instance_mask);
    }
    //end processing

//     Image display(confirmation)
//     printf("src",src_panel_imgs.size()
//    for (int j=0; j<src_panel_imgs.size(); j++) {
//        cv::imshow("panel", src_panel_imgs[j]);
//        cv::waitKey();
//    }
    return 0;
}
