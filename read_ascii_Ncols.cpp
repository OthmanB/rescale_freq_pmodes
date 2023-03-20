
#include <fstream>
#include <iostream>
#include <string>
#include <vector>
#include <Eigen/Dense>
#include "string_handler.h"
#include "data.h"
using Eigen::MatrixXd;

Data_file read_ascii_Ncols(const std::string file_in_name, const std::string delimiter, const bool verbose_data){
/*
 * This function read an input file (file_in_name) which may or may not contain a header + data in matricial form.
 * The number of column can be as small as 1
*/
    const bool verbose_steps=false;
    int cpt, Nrows;
    std::string line, line0, subline0; //token
    std::vector<std::string> header, data_str, tmp;
    long double tmp_val;
    std::vector<bool> isok(3);
    MatrixXd data;
    Data_file all_data_out; // The structure that encapsulate all the data, the header, the labels and units
    int data_Maxsize=1000;

    std::ifstream file_in;
    std::cout << "Reading the Data File..." << std::endl;
    std::cout << "  Assumptions for the data file: " << std::endl;
    std::cout << "       - header lines appear on the top of the file and are indicated by a # for first character. The header is however optional (then no # are found)" << std::endl;
    std::cout << "       - Maximum number of lines for the data: " << data_Maxsize << std::endl;
    file_in.open(file_in_name.c_str());
    if (file_in.is_open()) {
	    if(verbose_steps){
            std::cout << "...processing lines" << std::endl;
        }
		// [1] Get the header
		cpt=0;
		std::getline(file_in, line0);
		line0=strtrim(line0); // remove any white space at the begining/end of the string
		subline0=strtrim(line0.substr(0, 1)); // pick the first character
		subline0=subline0.c_str();
		if (subline0 == "#"){
			while(subline0 == "#"){		
				header.push_back(strtrim(line0.substr(1, std::string::npos))); // add all characters except the first one (and any space at the begining)

				std::getline(file_in, line0);
				line0=strtrim(line0); // remove any white space at the begining/end of the string
				subline0=strtrim(line0.substr(0, 1)); // pick the first character
				subline0=subline0.c_str();
				
				cpt=cpt+1;
			}
	        if(verbose_steps){
			    std::cout << "   [1] " << cpt << " header lines found..." << std::endl;
            }
		} else{
			header.push_back("");
			if(verbose_steps){
                std::cout << "   [1] Header not found. Header vector set to a blank vector<string> of size 1. Pursuing operations..." << std::endl;
            }
		}

        	if(verbose_steps){
                std::cout << "    [2] Rescaling parameters: Expecting one line for each of these with '=' separator..." << std::endl;
            }
		while (subline0 != "#"){ // We iterate on the parameters until we reach a comment symbol
            data_str=strsplit(strtrim(line0), "="); 
            if (data_str[0] == "Dnu_target"){
                all_data_out.Dnu_target=str_to_dbl(data_str[1]);
                if(verbose_data == true){
                    std::cout << "     - Dnu_target = " <<  all_data_out.Dnu_target << std::endl;
                }
                isok[0]=true;
            }
            if (data_str[0] == "epsilon_target"){
                all_data_out.epsilon_target=str_to_dbl(data_str[1]);
                if(verbose_data == true){
                    std::cout << "     - epsilon_target = " <<  all_data_out.epsilon_target << std::endl;
                }
                isok[1]=true;
            }
            if (data_str[0] == "d0l_target"){ 
                all_data_out.d0l_target =str_to_Xdarr(data_str[1], ",");
                if(verbose_data == true){
                    std::cout << "     - d0l_target = " <<  all_data_out.d0l_target.transpose() << std::endl;
                }
                isok[2]=true;
            }
            std::getline(file_in, line0);
		    subline0=strtrim(line0.substr(0, 1)); // pick the first character
		    subline0=subline0.c_str();
        }
        if (isok[0]*isok[1]*isok[2] == false){
            std::cerr << " Error : Missing expected argument for the target stars " << std::endl;
            std::cerr << "         Check that you have Dnu_target = [value]" << std::endl;
            std::cerr << "         Check that you have epsilon_target = [value]" << std::endl;
            std::cerr << "         Check that you have d0l_target = [value, value, value]" << std::endl;
            exit(EXIT_FAILURE);
        }
		// [2] Read the data...
        std::getline(file_in, line0); // Skip the comment line that we encountered
		if(verbose_steps){
            std::cout <<  "   [3] Reading the input data..." << std::endl;
        } 
		Nrows=0;
	    while(!file_in.eof()){
            data_str=strsplit(strtrim(line0), " \t"); 
            if (Nrows == 0) {
                data.resize(data_Maxsize, data_str.size());
            }
            for(int i=0; i<data_str.size();i++){
                if ( ! (std::istringstream(data_str[i]) >> tmp_val) ){
                        tmp_val = nan("");// If the number can be converted, then tmp_val=value. Otherwise tmp_val = NaN
                        std::cerr << " ERROR: IMPROPER VALUES FOUND IN INPUT FREQUENCY TABLE... REPLACED BY NAN: Check that there is no blank lines at the bottom of your input file" << std::endl; 
                        exit(EXIT_FAILURE);
                } 
                data(Nrows, i)=tmp_val;		
            }
            if (verbose_data == 1) {std::cout << data.row(Nrows) << std::endl;} // Show all entries only if requested
            std::getline(file_in, line0);
            Nrows=Nrows+1;
	    }
	file_in.close();
	data.conservativeResize(Nrows, data_str.size());
	std::cout << "         - Number of lines found: " << Nrows << std::endl;
	std::cout << "         - Number of columns found: " << data_str.size() << std::endl;
	std::cout << "      ----------------" << std::endl;
     } else{
        std::cerr << "Error: could not open input file " << file_in_name << std::endl;
        exit(EXIT_FAILURE);
    }
     all_data_out.data=data;
     all_data_out.header=header;

return all_data_out;
}
