
/* 
 *	Simple linear fit algorithm 
 * 
 *  Created on: 11 Oct 2017
 *      Author: obenomar
*/

# pragma once

# include <Eigen/Dense>
#include <iostream>
#include <cmath>

using Eigen::VectorXd;

VectorXd linfit(const VectorXd& x, const VectorXd& y);
