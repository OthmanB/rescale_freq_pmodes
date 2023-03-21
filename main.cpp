#include <iostream>
#include <fstream>
#include <Eigen/Dense>
#include <vector>
#include <iomanip>
#include "read_ascii_Ncols.h"
#include "data.h"
#include "rescale_freqs.h"

using namespace Eigen;
using Eigen::VectorXi;

int main(int argc, char** argv) {

    if (argc != 2) {
        std::cerr << "Usage: " << argv[0] << " <filename>" << std::endl;
        std::cerr << "   To rescale, the file must contain Dnu_target=[A value]\n epsilon_target=[A value]\n d0l_target=[val, val,val]" << std::endl;
        std::cerr << "   If these are not given, then it is assumed that you request a decomposition only" << std::endl;
        return 1;
    }
    const int Nchar = 20;
    const int precision=8;
    const bool verbose = false;
    const std::string delimiter = " \t";
    const char* filename = argv[1];
    double tol=1e-6;
    VectorXi ind;
    Data_file data;
    Freq_modes f_ref, f_rescaled;
    
    // Try to Open the input file
    std::ifstream infile(filename);
    if (!infile.is_open()) {
        std::cerr << "Error: could not open input file " << filename << std::endl;
        return 1;
    }
    // Read the file assuming a possible header + a matrix of data
    data=read_ascii_Ncols(filename, delimiter, verbose);
    // Define a compatible f_ref structure:
    ind=where_dbl(data.data.col(0), 0, tol);
    if(ind[0] != -1){
        f_ref.fl0.resize(ind.size());
        for(int i=0; i<ind.size();i++){
            f_ref.fl0[i]=data.data(ind[i],1);
        }
    } else{
        std::cerr << "ERROR : You must at least provide l=0 modes !" << std::endl;
        return 1;
    }
    // l=1
    ind=where_dbl(data.data.col(0), 1, tol);
    if(ind[0] != -1){
        f_ref.fl1.resize(ind.size());
        for(int i=0; i<ind.size();i++){
            f_ref.fl1[i]=data.data(ind[i],1);
        }
    } 
    // l=2
    ind=where_dbl(data.data.col(0), 2, tol);
    if(ind[0] != -1){
        f_ref.fl2.resize(ind.size());
        for(int i=0; i<ind.size();i++){
            f_ref.fl2[i]=data.data(ind[i],1);
        }
    } 
    // l=3
    ind=where_dbl(data.data.col(0), 3, tol);
    if(ind[0] != -1){
        f_ref.fl3.resize(ind.size());
        for(int i=0; i<ind.size();i++){
            f_ref.fl3[i]=data.data(ind[i],1);
        }
    } 
    // Rescale:
    if (data.do_rescale == true){
        f_rescaled=rescale_freqs(data.Dnu_target, data.epsilon_target, f_ref, data.d0l_target); // Rescale f_ref to match f_target asymptotic parameters
        if(f_rescaled.error_status == false){
            std::cout << "# Asymptotic parameters:" << std::endl;
            std::cout << " Dnu = " << data.Dnu_target << std::endl;
            std::cout << " epsilon = " << data.epsilon_target << std::endl;
            std::cout << " d0l = " << data.d0l_target.transpose() << std::endl;
            // Print the vector to the console
            std::cout << "# Individual frequencies:" << std::endl;
            std::cout << "# l " <<  std::setw(Nchar) <<  "nu" << std::endl;
            if (f_rescaled.fl0[0] !=-1){
                for(int i=0; i<f_rescaled.fl0.size();i++){
                    std::cout << "  0" << std::setprecision(precision) << std::setw(Nchar) << f_rescaled.fl0[i] <<   std::endl;
                }
            }
            if (f_rescaled.fl1[0] !=-1){
                for(int i=0; i<f_rescaled.fl1.size();i++){
                    std::cout << "  1" << std::setprecision(precision) << std::setw(Nchar) << f_rescaled.fl1[i] << std::endl;
                }
            }
            if (f_rescaled.fl2[0] !=-1){
                for(int i=0; i<f_rescaled.fl2.size();i++){
                    std::cout << "  2" << std::setprecision(precision) << std::setw(Nchar) << f_rescaled.fl2[i] << std::endl;
                }
            }
            if (f_rescaled.fl3[0] !=-1){
                for(int i=0; i<f_rescaled.fl3.size();i++){
                    std::cout << "  3" << std::setprecision(precision) << std::setw(Nchar) << f_rescaled.fl3[i] << std::endl;
                }
            }
        } else{
            std::cerr << "Error encountered when performing rescaling. Check your inputs" << std::endl;
            exit(EXIT_FAILURE);
        }
    } else{
        Data_asympt_p dec; // Result of the decomposition to be matched with the f_target.p_asympt parameters (except O2 that must match f_ref.p_asympt.O2_term*Dnu_ref/Dnu_star)
        dec=decompose_nu_nl(0, f_ref.fl0, f_ref.fl0, 0.25, false);
        if(dec.error_status == false){
            std::cout << "# Decomposition into asymptotic parameters:" << std::endl;
            std::cout << " Dnu = " << dec.Dnu << std::endl;
            std::cout << " epsilon = " << dec.epsilon << std::endl;
            std::cout << " # l" <<  std::setw(Nchar) <<  "nu" <<  std::setw(Nchar) <<   "d0l" <<  std::setw(Nchar) << "O2" << std::endl;
            if (f_ref.fl1.size()>0){
                if(dec.error_status == false){
                    dec=decompose_nu_nl(1, f_ref.fl0, f_ref.fl1, 0.25, false);
                } else{
                    std::cerr << "Error encountered when performing decomposition. Check your inputs" << std::endl;
                    exit(EXIT_FAILURE);                    
                }
                for(int i=0; i<f_ref.fl1.size();i++){
                    std::cout << "   1" << std::setprecision(precision) << std::setw(Nchar);
                    std::cout <<          std::setprecision(precision) << std::setw(Nchar) << f_ref.fl1[i];
                    std::cout <<          std::setprecision(precision) << std::setw(Nchar) << dec.d0l;
                    std::cout <<          std::setprecision(precision) << std::setw(Nchar) << dec.O2_term[i] << std::endl;
                }
            }
            if (f_ref.fl2.size()>0){
                if(dec.error_status == false){
                    dec=decompose_nu_nl(2, f_ref.fl0, f_ref.fl2, 0.25, false);
                } else{
                    std::cerr << "Error encountered when performing decomposition. Check your inputs" << std::endl;
                    exit(EXIT_FAILURE);                    
                }
                for(int i=0; i<f_ref.fl2.size();i++){
                    std::cout << "   2" << std::setprecision(precision) << std::setw(Nchar);
                    std::cout <<          std::setprecision(precision) << std::setw(Nchar) << f_ref.fl2[i];
                    std::cout <<          std::setprecision(precision) << std::setw(Nchar) << dec.d0l;
                    std::cout <<          std::setprecision(precision) << std::setw(Nchar) << dec.O2_term[i] << std::endl;
                }
            }
            if (f_ref.fl3.size()>0){
                if(dec.error_status == false){
                    dec=decompose_nu_nl(3, f_ref.fl0, f_ref.fl3, 0.25, false);
                } else{
                    std::cerr << "Error encountered when performing decomposition. Check your inputs" << std::endl;
                    exit(EXIT_FAILURE);                    
                }
                for(int i=0; i<f_ref.fl3.size();i++){
                    std::cout << "   3" << std::setprecision(precision) << std::setw(Nchar);
                    std::cout <<          std::setprecision(precision) << std::setw(Nchar) << f_ref.fl3[i];
                    std::cout <<          std::setprecision(precision) << std::setw(Nchar) << dec.d0l;
                    std::cout <<          std::setprecision(precision) << std::setw(Nchar) << dec.O2_term[i] << std::endl;
                }
            } 
        }
    }
    return 0;
}
