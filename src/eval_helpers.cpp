#include "read_file_path.hpp"
#include "twopage_to_onepage.hpp"
#include "page_classification.hpp"
#include <dirent.h>
#include <sys/stat.h>

// Simple implementation using POSIX directory functions for portability
static bool has_extension(const std::string &path, const std::string &ext) {
    if (ext.empty()) return true;
    std::string lower = path;
    std::string e = ext;
    for (auto &c : lower) c = std::tolower(c);
    for (auto &c : e) c = std::tolower(c);
    if (lower.size() >= e.size()) {
        return lower.substr(lower.size() - e.size()) == e;
    }
    return false;
}

std::vector<std::string> ReadFilePath::get_file_path(std::string dir_path) {
    std::vector<std::string> files;
    DIR *dp = opendir(dir_path.c_str());
    if (!dp) return files;
    struct dirent *entry;
    while ((entry = readdir(dp)) != NULL) {
        std::string name(entry->d_name);
        if (name == "." || name == "..") continue;
        std::string full = dir_path + "/" + name;
        struct stat st;
        if (stat(full.c_str(), &st) == 0 && S_ISREG(st.st_mode)) {
            files.push_back(full);
        }
    }
    closedir(dp);
    std::sort(files.begin(), files.end());
    return files;
}

std::vector<std::string> ReadFilePath::get_file_path(std::string dir_path, std::string extension) {
    std::vector<std::string> files;
    DIR *dp = opendir(dir_path.c_str());
    if (!dp) return files;
    struct dirent *entry;
    while ((entry = readdir(dp)) != NULL) {
        std::string name(entry->d_name);
        if (name == "." || name == "..") continue;
        std::string full = dir_path + "/" + name;
        struct stat st;
        if (stat(full.c_str(), &st) == 0 && S_ISREG(st.st_mode)) {
            if (has_extension(full, extension)) {
                files.push_back(full);
            }
        }
    }
    closedir(dp);
    std::sort(files.begin(), files.end());
    return files;
}

// PageCut: split if wide (two-page spread)
std::vector<cv::Mat> PageCut::pageCut(cv::Mat &input_page_image) {
    std::vector<cv::Mat> pages;
    int h = input_page_image.rows;
    int w = input_page_image.cols;
    if (w > h) {
        // split into left and right halves similar to Python version
        cv::Mat left = input_page_image(cv::Range::all(), cv::Range(0, w/2)).clone();
        cv::Mat right = input_page_image(cv::Range::all(), cv::Range(w/2, w)).clone();
        // return right then left to match original ordering
        pages.push_back(right);
        pages.push_back(left);
    } else {
        pages.push_back(input_page_image);
    }
    return pages;
}

// Very small heuristic for classification: detect mostly-dark pages as black pages
bool ClassificationPage::get_page_type(cv::Mat &src_image) {
    cv::Mat gray;
    if (src_image.channels() == 3) cv::cvtColor(src_image, gray, cv::COLOR_BGR2GRAY);
    else gray = src_image;
    // sample top 5 rows and bottom 5 rows and check if they are mostly zero
    int h = gray.rows;
    int w = gray.cols;
    int black_count = 0;
    int sample_pixels = 0;
    for (int y = 0; y < std::min(5, h); ++y) {
        for (int x = 0; x < w; ++x) {
            sample_pixels++;
            if (gray.at<uchar>(y, x) < 10) black_count++;
        }
    }
    for (int y = std::max(0, h-5); y < h; ++y) {
        for (int x = 0; x < w; ++x) {
            sample_pixels++;
            if (gray.at<uchar>(y, x) < 10) black_count++;
        }
    }
    double ratio = sample_pixels > 0 ? (double)black_count / sample_pixels : 0.0;
    return ratio > 0.9; // consider black page if >90% of sampled border pixels are dark
}
