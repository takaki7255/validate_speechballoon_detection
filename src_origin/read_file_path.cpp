#include "read_file_path.hpp"

std::vector<std::string> ReadFilePath::get_file_path(std::string dir_path){
    
    std::vector<std::string> file_paths;
    
    namespace fs = std::__fs::filesystem;
    
    // File existence check
    if (!fs::is_directory(dir_path)) {
        std::cout << "File does not exist." << std::endl;
        return file_paths;
    }
    
    for (const fs::directory_entry& file : fs::recursive_directory_iterator(dir_path)) {
        std::string path_name = file.path().string();// change type to string.
        file_paths.push_back(path_name);// add
    }
    std::sort(file_paths.begin(), file_paths.end());//sort filename
    return file_paths;
    
}

std::vector<std::string> ReadFilePath::get_file_path(std::string dir_path, std::string extension){
    
    std::vector<std::string> file_paths;
    
    namespace fs = std::__fs::filesystem;
    
    // File existence check
    if (!fs::is_directory(dir_path)) {
        std::cout << "File does not exist." << std::endl;
        return file_paths;
    }
    
    for (const fs::directory_entry& file : fs::recursive_directory_iterator(dir_path)) {
        std::string path_name = file.path().string();// change type to string.
        
        if (path_name.find(extension) != std::string::npos){
            file_paths.push_back(path_name);// add
        }
    }
    std::sort(file_paths.begin(), file_paths.end());//sort filename
    return file_paths;
    
}
