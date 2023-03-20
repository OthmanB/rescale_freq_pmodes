
#pragma once
#include <fstream>
#include <iostream>
#include <string>
#include <vector>
#include <Eigen/Dense>
#include "string_handler.h"
#include "data.h"
using Eigen::MatrixXd;

Data_file read_ascii_Ncols(const std::string file_in_name, const std::string delimiter, const bool verbose_data);