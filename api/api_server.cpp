#include <iostream>
#include <sstream>
#include <string>
#include <unordered_map>
#include <vector>
#include <Eigen/Dense>
#include "nlohmann/json.hpp"
#include "libs/cpp-httplib/httplib.h"

#include "../rescale_freqs.h"
#include "../string_handler.h"
#include "../decompose_nu.h"
#include "../linspace.h"
#include "../linfit.h"
#include "../data.h"

using json = nlohmann::json;
using Eigen::VectorXd;

/*
// Utility function to convert JSON to string
std::string jsonToString(const json& j) {
    std::ostringstream ss;
    ss << j;
    return ss.str();
}
*/
void handledata(const httplib::Request& req, httplib::Response& res) {
    Freq_modes f_ref, f_rescaled;
    Data_asympt_p dec; // Result of the decomposition to be matched with the f_target.p_asympt parameters (except O2 that must match f_ref.p_asympt.O2_term*Dnu_ref/Dnu_star)
        
    // Parse the JSON structure in the request body
    try {
        json data = json::parse(req.body);
        json response;
        // Translate into variables that can be handled by rescale_freqs() and decompose_nu_nl()
        VectorXd fl0 = Eigen::Map<VectorXd>(data["fl0"].get<std::vector<double>>().data(), data["fl0"].size()); // This is mandatory
        VectorXd fl1 = data["fl1"].empty() ? VectorXd() : Eigen::Map<VectorXd>(data["fl1"].get<std::vector<double>>().data(), data["fl1"].size());
        VectorXd fl2 = data["fl2"].empty() ? VectorXd() : Eigen::Map<VectorXd>(data["fl2"].get<std::vector<double>>().data(), data["fl2"].size());
        VectorXd fl3 = data["fl3"].empty() ? VectorXd() : Eigen::Map<VectorXd>(data["fl3"].get<std::vector<double>>().data(), data["fl3"].size());
        VectorXd d0l_target = data["d0l_target"].empty() ? VectorXd() : Eigen::Map<VectorXd>(data["d0l_target"].get<std::vector<double>>().data(), data["d0l_target"].size());
        f_ref.fl0=fl0;
        f_ref.fl1=fl1;
        f_ref.fl2=fl2;
        f_ref.fl3=fl3;
        if (data["do_rescale"].get<bool>() == true){
            f_rescaled=rescale_freqs(data["Dnu_target"].get<double>(), data["epsilon_target"].get<double>(), f_ref, d0l_target); // Rescale f_ref to match f_target asymptotic parameters
            // Send a success response along with data
            response = {
                {"status", "Data received successfully"},
                {"code", 200}
            };
            response["data"]["type"] = "Decomposition";
            response["data"]["result"]["error_status"] = f_rescaled.error_status;
            response["data"]["result"]["fl0"] = f_rescaled.fl0;
            response["data"]["result"]["fl1"] = f_rescaled.fl1;
            response["data"]["result"]["fl2"] = f_rescaled.fl2;
            response["data"]["result"]["fl3"] = f_rescaled.fl3;
        } else{
            dec=decompose_nu_nl(0, f_ref.fl0, f_ref.fl0, 0.25, false);
            if (f_ref.fl1.size()>0){
                dec=decompose_nu_nl(1, f_ref.fl0, f_ref.fl1, 0.25, false);
            }
            if (f_ref.fl2.size()>0){
                dec=decompose_nu_nl(2, f_ref.fl0, f_ref.fl2, 0.25, false);
            }
            if (f_ref.fl3.size()>0){
                dec=decompose_nu_nl(3, f_ref.fl0, f_ref.fl3, 0.25, false);
            }
            // Send a success response along with the data
             response ={
                {"status", "Data received successfully"},
                {"code", 200}
             };
            response["data"]["type"] = "Decomposition";
            response["data"]["result"]["error_status"] = dec.error_status;
            response["data"]["result"]["n"] = dec.n;
            response["data"]["result"]["Dnu"] = dec.Dnu;
            response["data"]["result"]["epsilon"] = dec.epsilon;
            response["data"]["result"]["d0l"] = dec.d0l;
            response["data"]["result"]["O2_term"] = dec.O2_term;
        }
 
        res.set_content(response.dump(), "application/json");
    } catch (json::exception& e) {
        // Invalid JSON structure, set the response status code to 400 Bad Request
        res.status = 400;
        // Construct an error message as JSON
        json error = {
            {"status", "Invalid JSON structure"},
            {"code", 400},
            {"error", e.what()}
        };
        res.set_content(error.dump(), "application/json");
    }
}

void run_api_server() {
    // Create HTTP server
    httplib::Server svr;
    std::cout << "Listening..." << std::endl;
    // Register endpoint handler functions
    svr.Post("/data", handledata);

    // Start server on port 8080
    svr.listen("localhost", 8080);
}

int main(){
    run_api_server();
}
