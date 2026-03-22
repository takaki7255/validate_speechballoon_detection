//
//  main_eval.cpp
//  Evaluation version of speechballoon detector
//
//  Modified to output evaluation masks instead of cropped images
//

#include <stdio.h>
#include <iostream>
#include <opencv2/opencv.hpp>
#include <vector>
#include <sstream>
#include <iomanip>
#include <sys/stat.h>
//headerfile
#include "read_file_path.hpp"
#include "twopage_to_onepage.hpp"
#include "page_classification.hpp"
#include "frame_separation.hpp"
#include "speechballoon_separation.hpp"

// Helper function to create directories
void create_directory(const std::string& path) {
    mkdir(path.c_str(), 0755);
}

int main(int argc, const char * argv[]) {
    // Accept command line arguments for input/output paths
    std::string input_folder = (argc > 1) ? argv[1] : "evaluation_dataset/sampled_images";
    std::string output_base = (argc > 2) ? argv[2] : "predictions_origin";
    
    // Create output directories
    create_directory(output_base);
    std::string eval_masks_dir = output_base + "/eval_masks";
    std::string eval_instances_dir = output_base + "/eval_instances";
    create_directory(eval_masks_dir);
    create_directory(eval_instances_dir);
    
    std::cout << "Input folder: " << input_folder << std::endl;
    std::cout << "Output folder: " << output_base << std::endl;
    
    //Load images in a folder
    std::vector<std::string> image_paths;
    image_paths = ReadFilePath::get_file_path(input_folder, "jpg");
    if (image_paths.empty()) {
        image_paths = ReadFilePath::get_file_path(input_folder, "png");
    }
    
    std::cout << "Found " << image_paths.size() << " images" << std::endl;

    //Variable Definition
    std::vector<cv::Mat> page_img;//A group of page images(spread cut out of the image)
    //--frame extraction
    Framedetect panel;//instantiate
    //--speechballoon extraction
    Speechballoon speechballoon;//instantiate
    static int speechballoon_max = 99;//Specify maximum number of pieces
    
    //Start processing
    for (int i = 0; i < image_paths.size(); i++) {
        bool verbose = (i < 3); // Debug first 3 images only
        
        // Extract filename without extension
        std::string full_path = image_paths[i];
        size_t last_slash = full_path.find_last_of("/\\");
        size_t last_dot = full_path.find_last_of(".");
        std::string filename = full_path.substr(last_slash + 1, last_dot - last_slash - 1);
        
        // Load original image (for size reference)
        cv::Mat orig_bgr = cv::imread(image_paths[i], cv::IMREAD_COLOR);
        if (orig_bgr.empty()) continue;
        
        cv::Mat img = cv::imread(image_paths[i], 0);//grayscale loading
        if (img.empty()) continue;
        
        int orig_h = orig_bgr.rows;
        int orig_w = orig_bgr.cols;
        
        std::cout << "[" << i+1 << "/" << image_paths.size() << "] " << filename << std::endl;
        if (verbose) {
            std::cout << "  Image size: " << orig_w << "x" << orig_h << std::endl;
        }
        
        // Create output masks (same size as original image)
        cv::Mat binary_mask = cv::Mat::zeros(orig_h, orig_w, CV_8UC1);
        cv::Mat instance_mask = cv::Mat::zeros(orig_h, orig_w, CV_8UC1);
        int instance_id = 1;
        
        page_img = PageCut::pageCut(img);//two pages apart
        
        //---start Processing on page---//
        for (int j = 0; j < page_img.size(); j++) {
            // Calculate page offset
            // PageCut returns [right_page, left_page] for two-page spreads
            int page_offset_x = (j == 0) ? orig_w / 2 : 0;
            
            if (verbose) {
                std::cout << "  Page " << j << ": offset_x=" << page_offset_x 
                          << ", size=" << page_img[j].cols << "x" << page_img[j].rows << std::endl;
            }
            
            //*page classification
            bool is_black_page = ClassificationPage::get_page_type(page_img[j]);//0-white 1-black
            if (is_black_page) {
                std::cout << "  Page " << j << " is black, skipping" << std::endl;
                continue;
            }
            
            // Create page-size masks (0 initialized)
            cv::Mat page_sem_mask = cv::Mat::zeros(page_img[j].rows, page_img[j].cols, CV_8UC1);
            cv::Mat page_ins_mask = cv::Mat::zeros(page_img[j].rows, page_img[j].cols, CV_8UC1);
            int page_instance_counter = 0;
            
            //*frame extraction with bbox
            std::vector<Panel> panels = panel.frame_detect_with_bbox(page_img[j]);
            
            if (verbose) {
                std::cout << "    Found " << panels.size() << " panels" << std::endl;
            }
            
            //---start Processing on panel---//
            for (size_t k = 0; k < panels.size(); k++) {
                Panel& p = panels[k];
                
                if (verbose) {
                    std::cout << "    Panel[" << k << "]: bbox=(" << p.bbox.x << "," << p.bbox.y 
                              << "," << p.bbox.width << "," << p.bbox.height << ")" << std::endl;
                }
                
                //*speechballoon extraction with bbox and mask
                std::vector<Balloon> balloons = speechballoon.speechballoon_detect_with_bbox(p.img);
                
                if (verbose) {
                    std::cout << "      Found " << balloons.size() << " balloons" << std::endl;
                }
                
                if (balloons.empty()) continue;
                if (balloons.size() > speechballoon_max) continue;
                
                for (size_t index = 0; index < balloons.size(); index++) {
                    const Balloon& b = balloons[index];
                    
                    if (b.mask.empty()) continue;
                    
                    // Debug: print balloon coordinates
                    if (verbose) {
                        std::cout << "    Panel[" << k << "] Balloon[" << index << "]: "
                                  << "bbox in panel=(" << b.bbox.x << "," << b.bbox.y 
                                  << "," << b.bbox.width << "," << b.bbox.height << ") "
                                  << "panel_bbox=(" << p.bbox.x << "," << p.bbox.y 
                                  << "," << p.bbox.width << "," << p.bbox.height << ") "
                                  << "page_offset_x=" << page_offset_x << std::endl;
                    }
                    
                    // Extract contours from the mask
                    std::vector<std::vector<cv::Point>> contours;
                    cv::findContours(b.mask.clone(), contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);
                    
                    if (verbose) {
                        std::cout << "      Found " << contours.size() << " contours" << std::endl;
                    }
                    
                    page_instance_counter++;
                    int instance_id_local = page_instance_counter;
                    
                    // Transform contours to page coordinates
                    for (auto& contour : contours) {
                        if (contour.size() < 3) continue;
                        
                        // Debug: print first point before and after transformation
                        if (verbose && !contour.empty()) {
                            cv::Point original = contour[0];
                            cv::Point transformed(
                                original.x + b.bbox.x + p.bbox.x,
                                original.y + b.bbox.y + p.bbox.y
                            );
                            std::cout << "      Contour[0] point: (" << original.x << "," << original.y << ") -> "
                                      << "page coords: (" << transformed.x << "," << transformed.y << ") -> "
                                      << "final coords: (" << (transformed.x + page_offset_x) << "," << transformed.y << ")" << std::endl;
                        }
                        
                        // Offset contour points by balloon bbox (relative to panel) + panel bbox (relative to page)
                        std::vector<cv::Point> offset_contour;
                        for (const auto& pt : contour) {
                            offset_contour.push_back(cv::Point(
                                pt.x + b.bbox.x + p.bbox.x,
                                pt.y + b.bbox.y + p.bbox.y
                            ));
                        }
                        
                        // Draw on page masks
                        cv::drawContours(page_sem_mask, std::vector<std::vector<cv::Point>>{offset_contour},
                                       -1, cv::Scalar(255), cv::FILLED);
                        cv::drawContours(page_ins_mask, std::vector<std::vector<cv::Point>>{offset_contour},
                                       -1, cv::Scalar(instance_id_local), cv::FILLED);
                    }
                }
            }
            //---end Processing on panel---//
            
            // Copy page masks to original image coordinates
            cv::Rect page_roi(page_offset_x, 0, page_img[j].cols, page_img[j].rows);
            
            // Semantic: copy directly
            page_sem_mask.copyTo(binary_mask(page_roi));
            
            // Instance: map IDs and copy
            for (int y = 0; y < page_ins_mask.rows; y++) {
                for (int x = 0; x < page_ins_mask.cols; x++) {
                    uint8_t local_id = page_ins_mask.at<uint8_t>(y, x);
                    if (local_id > 0) {
                        int global_id = instance_id + local_id;
                        if (global_id <= 254) {
                            instance_mask.at<uint8_t>(y + page_roi.y, x + page_roi.x) = (uint8_t)global_id;
                        }
                    }
                }
            }
            
            instance_id += page_instance_counter;
        }
        //---end Processing on page---//
        
        // Save masks
        std::string mask_path = eval_masks_dir + "/" + filename + ".png";
        std::string instance_path = eval_instances_dir + "/" + filename + ".png";
        
        cv::imwrite(mask_path, binary_mask);
        cv::imwrite(instance_path, instance_mask);
        
        std::cout << "  Saved masks with " << (instance_id - 1) << " instances" << std::endl;
    }
    //end processing
    
    std::cout << "Processing complete!" << std::endl;
    return 0;
}
