#ifndef read_file_path_hpp
#define read_file_path_hpp

#include <stdio.h>
#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <filesystem>

class ReadFilePath {
public:
    static std::vector<std::string> get_file_path(std::string dir_path); //Read all files in the directory
    
    static std::vector<std::string> get_file_path(std::string dir_path, std::string extension); // Read all files in the directory(*extension ex>.png .jpg)
private:
};

#endif /* read_file_path_hpp */
