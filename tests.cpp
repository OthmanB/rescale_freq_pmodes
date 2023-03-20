#include <Eigen/Dense>
#include <cmath>
#include <random>
//#include <sstream>
//#include <iostream>
#include <iomanip>
#include "linspace.h"
#include "linfit.h"
#include "decompose_nu.h"
#include "rescale_freqs.h"

using Eigen::VectorXd;

Freq_modes load_test_data(void);
void test_decompose(void);
void test_rescale(void);

Data_asympt_p gen_asymp_data(){
    /*
        Create random test data following the p modes asymptotic relation
        Cover a range typical for a MS star:
             n0 ~ [8, 14]
             Dnu ~ [20, 200]
             epsilon ~ [0,1]
             d0l ~ [-0.05*Dnu, +0.05*Dnu]
             O2 ~ [-0.02*Dnu, +0.02*Dnu]
    */    
    const int Nmodes=20;
    int n0;
    long double O2m;
    long double Dnu, epsilon, d0l;
    VectorXd O2(Nmodes), n(Nmodes);
    Data_asympt_p asymp;

   	std::random_device rd;
	std::mt19937 gen(rd()); //Standard mersenne_twister_engine seeded with rd()
    std::normal_distribution<double> distrib_normal(0.,1.);
	std::uniform_real_distribution<double> distrib(0,1);
    std::uniform_int_distribution<int> distrib_discrete(8,14);
	n0=distrib_discrete(gen); // [8, 14]
    n=linspace(n0,n0+Nmodes-1, Nmodes);
    Dnu=distrib(gen)*180 + 20; // [20, 200]
    epsilon=distrib(gen);

   d0l=(distrib(gen) - 0.5)*2*Dnu*5/100; // [-0.05*Dnu, +0.05*Dnu]
   for(int i=0; i<O2.size(); i++){
        /*
        r=distrib_normal(gen)*Dnu/1000.;
        while (r>Dnu/20){            
           r=distrib_normal(gen)*Dnu/1000.;
        }
       */
       O2[i]=0.008*(n[i]-n.mean())*(n[i]-n.mean())/2;
       //std::cout << "O2[i] = " << O2[i] << std::endl;
    }
    // O2 is a second order effect. Thus it must be of 0-mean. Otherwise, the decomposition 
    // with decompose_nu_nl() will shift systematically d0l and O2
    O2m=O2.mean();
    for(int i=0; i<O2.size();i++){
        O2[i]=O2[i] - O2m;
    }
    asymp.n=n;
    asymp.Dnu=Dnu;
    asymp.epsilon=epsilon;
    asymp.d0l=d0l;
    asymp.O2_term=O2;
    return asymp;
}
Freq_modes gen_freq_data(){
    /*
        use gen_asymp_data() in order to create frequencies that 
        are used for the reference star
    */
    int l;
    Data_asympt_p asymp=gen_asymp_data();
    Freq_modes nus;
    VectorXd cte(asymp.n.size());
    l=0;
    cte.setConstant((asymp.epsilon)*asymp.Dnu);
    nus.fl0=cte + asymp.n*asymp.Dnu + asymp.O2_term;
    l=1;
    cte.setConstant((asymp.epsilon + 1.*l/2)*asymp.Dnu + asymp.d0l);
    nus.fl1=cte + asymp.n*asymp.Dnu + asymp.O2_term;
    l=2;
    cte.setConstant((asymp.epsilon + 1.*l/2)*asymp.Dnu + asymp.d0l);
    nus.fl2=cte + asymp.n*asymp.Dnu + asymp.O2_term;
    l=3;
    cte.setConstant((asymp.epsilon + 1.*l/2)*asymp.Dnu + asymp.d0l);
    nus.fl3=cte + asymp.n*asymp.Dnu + asymp.O2_term;

    nus.p_asymptotic.n=asymp.n;
    nus.p_asymptotic.Dnu=asymp.Dnu;
    nus.p_asymptotic.epsilon=asymp.epsilon;
    nus.p_asymptotic.d0l=asymp.d0l;
    nus.p_asymptotic.O2_term=asymp.O2_term;
    return nus;
}

bool check_deltas(const Data_asympt_p asymp_target, const VectorXd& fl0_rescaled, const VectorXd& fl_rescaled, const int l, 
                    const double Cfactor, const double tol, const bool verbose){
    bool ok=true;
    long double ratio_mean;
    Data_asympt_p out; // Result of the decomposition to be matched with the f_target.p_asympt parameters (except O2 that must match f_ref.p_asympt.O2_term*Dnu_ref/Dnu_star)
    out=decompose_nu_nl(l, fl0_rescaled, fl_rescaled, Cfactor, false);
    if (std::abs(asymp_target.Dnu-out.Dnu) >= tol || std::abs(asymp_target.epsilon- out.epsilon) >= tol){
        ok=false;
    } 
    if (l !=0 && std::abs(asymp_target.d0l-out.d0l) >= tol){
        ok=false;
    }
    for(int i=0; i<out.O2_term.size();i++){
        if (std::abs(asymp_target.O2_term[i]/out.O2_term[i]) >= (1. + tol) ){
            ok=false;
        }
    }
   if (verbose == true){
        std::cout << " Dnu_in     : " << asymp_target.Dnu << "   Dnu    : " << out.Dnu << "   delta: " << asymp_target.Dnu-out.Dnu  << std::endl;
        std::cout << " epsilon_in : " <<  asymp_target.epsilon   << "  epsilon: " << out.epsilon  << " delta: " << asymp_target.epsilon- out.epsilon << std::endl;
        if (l !=0){
            std::cout << " d0l_in     : " << asymp_target.d0l  << "    d0l    :  " << out.d0l << "   delta: " << asymp_target.d0l-out.d0l << std::endl;
        } else{
            std::cout << " d0l_in     : " << 0 << "     d0l    :  " << out.d0l << "   delta: " << 0-out.d0l << std::endl;
        }
        std::cout << " O2l_in     : "  << std::endl;
        std::cout << "     " <<  asymp_target.O2_term.transpose()  << std::endl;
        std::cout << " new O2l: "  << std::endl;
        std::cout << "     " <<  out.O2_term.transpose()  << std::endl;
        std::cout << "target: n.size() =" << asymp_target.n.size() << "       n0: " << asymp_target.n[0] << std::endl;
        std::cout << "out   : n.size() =" << out.n.size() << "       n0: " << out.n[0] << std::endl;
        //std::cout << " n(target)    n(decompose)    O2(target)/O2(decompose): " << std::endl;        
        ratio_mean=0;
        for(int i=0; i<out.O2_term.size();i++){
            std::cout << asymp_target.n[i]  << "   " << out.n[i] << "   "  << asymp_target.O2_term[i]/out.O2_term[i] << std::endl;
            ratio_mean=ratio_mean + asymp_target.O2_term[i]/out.O2_term[i];
        }
        ratio_mean=ratio_mean/out.O2_term.size();
        std::cout << "mean(O2(target)/O2(decompose)) = " << ratio_mean << std::endl;
        //std::cout << "Note: O2(target)/O2(decompose) MUST close to one. Otherwise, there is an issue" << std::endl;
        std::cout << "---" << std::endl;
    }
        
    return ok;
}

Freq_modes load_test_data(void){
    /*
        These data are frequencies for 16 Cyg A
    */
    Freq_modes f;
    f.fl0.resize(18); f.fl1.resize(17); f.fl2.resize(18); f.fl3.resize(14);
    f.fl0 << 1287.7378, 1391.6472, 1494.9822, 1598.7040, 1700.9117, 1802.3264,1904.5894, 2007.5698,
        2110.8952, 2214.2624, 2317.2959, 2420.9247, 2524.9409, 2629.3219, 2734.3085, 2839.1342,2944.6692, 3048.5000;
    f.fl1 << 1334.1481, 1437.7227, 1541.9739, 1644.9908, 1747.1738, 1849.0120, 1952.0236, 2055.4881, 2159.1390, 2262.5176,
            2366.1802, 2470.2896, 2574.7077, 2679.4912, 2784.2952,  2890.0152, 2995.1098;
    f.fl2 << 1279.5453, 1383.9922, 1487.7844,  1591.2323,  1694.0395, 1795.8284, 1898.3264, 2001.7400,  2105.3332,
            2208.9378,  2312.4974,  2416.3217,   2520.5174,  2624.8247,  2730.0274, 2836.0081,  2942.1468,  3046.5000;
    f.fl3 << 1427.44580, 1530.34899, 1632.59738, 1735.49943, 1838.42350, 1941.60645, 2045.80496, 2149.91448,
            2253.57221, 2357.48682, 2461.82262, 2566.53772, 2670.15963, 2774.16196;
    return f;
}
void test_decompose(void){
    // Values can be compared with the Python implementation in the python directory
    Freq_modes f;
    Data_asympt_p output;
    const bool verbose = true;
    const double Cfactor=0.25;
    f=load_test_data();
    std::cout << "    ----  l = 0 ----" << std::endl;
    output=decompose_nu_nl(0, f.fl0, f.fl0, Cfactor, verbose);
    std::cout << "    ----  l = 1 ----" << std::endl;
    output=decompose_nu_nl(1, f.fl0, f.fl1, Cfactor, verbose);
    std::cout << "    ----  l = 2 ----" << std::endl;
    output=decompose_nu_nl(2, f.fl0, f.fl2, Cfactor, verbose);
    std::cout << "    ----  l = 3 ----" << std::endl;
    output=decompose_nu_nl(3, f.fl0, f.fl3, Cfactor, verbose);
}

void test_consistency(void){
    /*
        Run a test that check if when you put fl in input:
            - decompose_nu_nl() ensure that fl=(n+epsilon+l/2)*Dnu + d0l + O2 is EXACTLY identifcal to the input fl
            - rescale_freqs() with executed with no rescaling ensure that fl as output is EXACTLY fl in input
    */
    const double tol=0.0001;
    const double Dnu_star = 80.;
    const double epsilon_star = 0.5;
    bool status=true;
    double ftmp;
    VectorXd d0l_star(3);
    d0l_star << -1.5, -1.5,-1.5;
    Freq_modes f_ref, f_star;
    Data_asympt_p asymp_target;
    //f_ref=load_test_data();
    f_ref=gen_freq_data();
    std::cout << " --- Test that we can reconstruct f_ref solely from p_asymptotic ---" << std::endl;
    std::cout << " --- l=0 ----" << std::endl;
    asymp_target=decompose_nu_nl(0, f_ref.fl0, f_ref.fl0, 0.25, false);
    std::cout << std::setw(15) << "fl" << std::setw(15) << "asymp + O2" << std::setw(15) << "delta" << std::setw(12) << "n(f_ref)" << std::setw(15) << "espilon(f_ref)" << std::setw(12) << "Dnu(f_ref)" << std::setw(12) << "O2(f_ref)" << std::setw(12) << "n(decomp)" << std::setw(15) << "espilon(decomp)" << std::setw(12) << "Dnu(decomp)" << std::setw(12) << "O2(decomp)" << std::endl;
    for(int i=0; i<f_ref.p_asymptotic.n.size();i++){
        ftmp=(asymp_target.n[i] + asymp_target.epsilon)*asymp_target.Dnu + asymp_target.O2_term[i];
        std::cout << std::setw(15) << f_ref.fl0[i] << std::setw(15) << ftmp << std::setw(15) << f_ref.fl0[i] - ftmp  <<  std::setw(15) << f_ref.p_asymptotic.n[i] <<  std::setw(12) << f_ref.p_asymptotic.epsilon <<  std::setw(12) << f_ref.p_asymptotic.Dnu <<  std::setw(12) << f_ref.p_asymptotic.O2_term[i] << std::setw(12) << asymp_target.n[i] <<  std::setw(12) << asymp_target.epsilon <<  std::setw(12) << asymp_target.Dnu <<  std::setw(12) << asymp_target.O2_term[i] << std::endl;
        if ((f_ref.fl0[i] - ftmp) >= tol){
            status=false;
        }
    }
    std::cout << " --- l=1 ----" << std::endl;
    asymp_target=decompose_nu_nl(1, f_ref.fl0, f_ref.fl1, 0.25, false);
    for(int i=0; i<f_ref.p_asymptotic.n.size();i++){
        ftmp=(asymp_target.n[i] + asymp_target.epsilon + 0.5)*asymp_target.Dnu + f_ref.p_asymptotic.d0l + asymp_target.O2_term[i];
        std::cout << std::setw(15) << f_ref.fl1[i] << std::setw(15) << ftmp << std::setw(15) << f_ref.fl1[i] - ftmp  <<  std::setw(15) << f_ref.p_asymptotic.n[i] <<  std::setw(12) << f_ref.p_asymptotic.epsilon <<  std::setw(12) << f_ref.p_asymptotic.Dnu <<  std::setw(12) << f_ref.p_asymptotic.O2_term[i] << std::setw(12) << asymp_target.n[i] <<  std::setw(12) << asymp_target.epsilon <<  std::setw(12) << asymp_target.Dnu <<  std::setw(12) << asymp_target.O2_term[i] << std::endl;
        if ((f_ref.fl1[i] - ftmp) >= tol){
            status=false;
        }
    }
    std::cout << " --- l=2 ----" << std::endl;
    asymp_target=decompose_nu_nl(2, f_ref.fl0, f_ref.fl2, 0.25, false);
    for(int i=0; i<f_ref.p_asymptotic.n.size();i++){
        ftmp=(asymp_target.n[i] + asymp_target.epsilon + 1)*asymp_target.Dnu + f_ref.p_asymptotic.d0l + asymp_target.O2_term[i];
        std::cout << std::setw(15) << f_ref.fl2[i] << std::setw(15) << ftmp << std::setw(15) << f_ref.fl2[i] - ftmp  <<  std::setw(15) << f_ref.p_asymptotic.n[i] <<  std::setw(12) << f_ref.p_asymptotic.epsilon <<  std::setw(12) << f_ref.p_asymptotic.Dnu <<  std::setw(12) << f_ref.p_asymptotic.O2_term[i] << std::setw(12) << asymp_target.n[i] <<  std::setw(12) << asymp_target.epsilon <<  std::setw(12) << asymp_target.Dnu <<  std::setw(12) << asymp_target.O2_term[i] << std::endl;
        if ((f_ref.fl2[i] - ftmp) >= tol){
            status=false;
        }   
    }
    std::cout << " --- l=3 ----" << std::endl;
    asymp_target=decompose_nu_nl(3, f_ref.fl0, f_ref.fl3, 0.25, false);
    for(int i=0; i<f_ref.p_asymptotic.n.size();i++){
        ftmp=(asymp_target.n[i] + asymp_target.epsilon + 1.5)*asymp_target.Dnu + f_ref.p_asymptotic.d0l + asymp_target.O2_term[i];
        std::cout << std::setw(15) << f_ref.fl3[i] << std::setw(15) << ftmp << std::setw(15) << f_ref.fl3[i] - ftmp  <<  std::setw(15) << f_ref.p_asymptotic.n[i] <<  std::setw(12) << f_ref.p_asymptotic.epsilon <<  std::setw(12) << f_ref.p_asymptotic.Dnu <<  std::setw(12) << f_ref.p_asymptotic.O2_term[i] << std::setw(12) << asymp_target.n[i] <<  std::setw(12) << asymp_target.epsilon <<  std::setw(12) << asymp_target.Dnu <<  std::setw(12) << asymp_target.O2_term[i] << std::endl;
        if ((f_ref.fl3[i] - ftmp) >= tol){
            status=false;
        }   
    }
    if (status == false){
        std::cout << " ERROR: Element detected beyond tolerance!" << std::endl;
        std::cout << "        Debug required" << std::endl;
        exit(EXIT_FAILURE);
    }  else{
        std::cout << "---------------------------------" << std::endl;
        std::cout << "              ALL OK             " << std::endl;
        std::cout << "---------------------------------" << std::endl;
    }      
    std::cout << " ---- " << std::endl;

   std::cout << " --- Test that the rescale_freqs() function is self-consisent ---" << std::endl;
 
    asymp_target.Dnu=f_ref.p_asymptotic.Dnu;
    asymp_target.epsilon=f_ref.p_asymptotic.epsilon;
    asymp_target.d0l=f_ref.p_asymptotic.d0l;
    asymp_target.n=f_ref.p_asymptotic.n;
    asymp_target.O2_term=f_ref.p_asymptotic.O2_term;
    d0l_star << f_ref.p_asymptotic.d0l, f_ref.p_asymptotic.d0l, f_ref.p_asymptotic.d0l;
    f_star=rescale_freqs(f_ref.p_asymptotic.Dnu, f_ref.p_asymptotic.epsilon, f_ref, d0l_star);

    std::cout << " --- l=0 ----" << std::endl;
    status=check_deltas(asymp_target, f_star.fl0, f_star.fl0, 0, 0.25, tol, true);
    std::cout << " --- l=1 ----" << std::endl;
    status=check_deltas(asymp_target, f_star.fl0, f_star.fl1, 1, 0.25, tol, true);
    std::cout << " --- l=2 ----" << std::endl;
    status=check_deltas(asymp_target, f_star.fl0, f_star.fl2, 2, 0.25, tol, true);
    std::cout << " --- l=3 ----" << std::endl;
    status=check_deltas(asymp_target, f_star.fl0, f_star.fl3, 3, 0.25, tol, true);
    if (status == false){
        std::cout << " ERROR: Element detected beyond tolerance!" << std::endl;
        std::cout << "        Debug required" << std::endl;
        exit(EXIT_FAILURE);
    }  else{
        std::cout << "---------------------------------" << std::endl;
        std::cout << "              ALL OK             " << std::endl;
        std::cout << "---------------------------------" << std::endl;
    }      

}

void test_thorough(void){
    /*
        A thorough test that scan a range of possible parameters
        and check that the rescaling works as intended
    */
    const double tol=0.001; // Maximum admissible difference between the rescaled frequencies and the target frequencies
    const int Niter=100; // How many times we generate random set of frequencies
    const bool verbose = false;
    const double Cfactor=0.25;
    bool status;
    VectorXd d0l_star(3);
    Data_asympt_p asymp_target;
    Freq_modes f_ref, f_rescaled;
    for(int i=0;i<Niter;i++){
        f_ref=gen_freq_data();
        /*
        std::cout << " --- f_ref.p_asymptotic ---" << std::endl;
        std::cout << " f_ref.p_asymptotic.Dnu : " << f_ref.p_asymptotic.Dnu << std::endl;
        std::cout << " f_ref.p_asymptotic.epsilon : " << f_ref.p_asymptotic.epsilon << std::endl;
        std::cout << " f_ref.p_asymptotic.d0l : " << f_ref.p_asymptotic.d0l << std::endl;
        std::cout << " f_ref.p_asymptotic.O2_term : " << f_ref.p_asymptotic.O2_term.transpose() << std::endl;
        std::cout << " f_ref.p_asymptotic.n: " << f_ref.p_asymptotic.n.transpose() << std::endl;
        std::cout << " --- " << std::endl; 
        std::cout << " --- " << std::endl; 
        */
        //std::cout << "    - - - - - l=0 - - - - " << std::endl;
        status=check_deltas(f_ref.p_asymptotic, f_ref.fl0, f_ref.fl0, 0, Cfactor, tol, verbose); // Check that the reference has no issue
        //std::cout << "    - - - - - l=1 - - - - " << std::endl;
        status=check_deltas(f_ref.p_asymptotic, f_ref.fl0, f_ref.fl1, 1, Cfactor, tol, verbose); // Check that the reference has no issue
        //std::cout << "    - - - - - l=2 - - - - " << std::endl;
        status=check_deltas(f_ref.p_asymptotic, f_ref.fl0, f_ref.fl2, 2, Cfactor, tol, verbose); // Check that the reference has no issue
        //std::cout << "    - - - - - l=3 - - - - " << std::endl;
        status=check_deltas(f_ref.p_asymptotic, f_ref.fl0, f_ref.fl3, 3, Cfactor, tol, verbose); // Check that the reference has no issue
        asymp_target=gen_asymp_data(); // Generate a random set of values for the asymptotic
        asymp_target.n=f_ref.p_asymptotic.n; // set n to be the same in the target and in the ref
        asymp_target.O2_term=f_ref.p_asymptotic.O2_term*asymp_target.Dnu/f_ref.p_asymptotic.Dnu; // set O2_term to be the rescaled one of ref
        d0l_star << asymp_target.d0l, asymp_target.d0l, asymp_target.d0l; // set l=1,2,3 d0l to be the same, for simplicity
        f_rescaled=rescale_freqs(asymp_target.Dnu, asymp_target.epsilon, f_ref, d0l_star); // Rescale f_ref to match f_target asymptotic parameters
        //std::cout << "    - - - - - l=0 - - - - " << std::endl;
        status=check_deltas(asymp_target, f_rescaled.fl0, f_rescaled.fl0, 0, Cfactor, tol, verbose);
        if (status == false){
            std::cout << " ERROR: Element detected beyond tolerance!" << std::endl;
            std::cout << "        Debug required" << std::endl;
            exit(EXIT_FAILURE);
        }  else{
            std::cout << "---------------------------------" << std::endl;
            std::cout << "l=0                OK  ("<< i+1 << "/" << Niter << ")              " << std::endl;
            std::cout << "---------------------------------" << std::endl;
        }    
        //std::cout << "    - - - - - l=1 - - - - " << std::endl;
        status=check_deltas(asymp_target, f_rescaled.fl0, f_rescaled.fl1, 1, Cfactor, tol, verbose);
        if (status == false){
            std::cout << " ERROR: Element detected beyond tolerance!" << std::endl;
            std::cout << "        Debug required" << std::endl;
            exit(EXIT_FAILURE);
        }  else{
            std::cout << "---------------------------------" << std::endl;
            std::cout << "l=1                OK  ("<< i+1 << "/" << Niter << ")              " << std::endl;
            std::cout << "---------------------------------" << std::endl;
        }         
        //std::cout << "    - - - - - l=2 - - - - " << std::endl;
        status=check_deltas(asymp_target, f_rescaled.fl0, f_rescaled.fl2, 2, Cfactor, tol, verbose);
        if (status == false){
            std::cout << " ERROR: Element detected beyond tolerance!" << std::endl;
            std::cout << "        Debug required" << std::endl;
            exit(EXIT_FAILURE);
        }  else{
            std::cout << "---------------------------------" << std::endl;
            std::cout << "l=2               OK  ("<< i+1 << "/" << Niter << ")              " << std::endl;
            std::cout << "---------------------------------" << std::endl;
        }        
        //std::cout << "    - - - - - l=3 - - - - " << std::endl;
        status=check_deltas(asymp_target, f_rescaled.fl0, f_rescaled.fl3, 3, Cfactor, tol, verbose);
        if (status == false){
            std::cout << " ERROR: Element detected beyond tolerance!" << std::endl;
            std::cout << "        Debug required" << std::endl;
            exit(EXIT_FAILURE);
        }  else{
            std::cout << "---------------------------------" << std::endl;
            std::cout << "l=3                OK  ("<< i+1 << "/" << Niter << ")              " << std::endl;
            std::cout << "---------------------------------" << std::endl;
        }      
    }
}

int main(void){
    //test_decompose();
    std::cout << "---------------------------------" << std::endl;
    std::cout << "---------------------------------" << std::endl;
    std::cout << "  Checks with test_consistency() " << std::endl;
    std::cout << "---------------------------------" << std::endl;
    std::cout << "---------------------------------" << std::endl;
    test_consistency();
    std::cout << "---------------------------------" << std::endl;
    std::cout << "---------------------------------" << std::endl;
    std::cout << "  Checks with test_thorough() " << std::endl;
    std::cout << "---------------------------------" << std::endl;
    std::cout << "---------------------------------" << std::endl;
   test_thorough();
}