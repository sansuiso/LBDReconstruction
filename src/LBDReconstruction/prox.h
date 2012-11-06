//  prox.h
//
//	Copyright (C) 2011-2012  Signal Processing Laboratory 2 (LTS2), EPFL,
//	Emmanuel d'Angelo (emmanuel.dangelo@epfl.ch),
//	Laurent Jacques (laurent.jacques@uclouvain.be)
//	Alexandre Alahi (alahi@stanford.edu)
//	and Pierre Vandergheynst (pierre.vandergheynst@epfl.ch)
//
//  Redistribution and use in source and binary forms, with or without modification,
//  are permitted provided that the following conditions are met:
//
//   * Redistribution's of source code must retain the above copyright notice,
//     this list of conditions and the following disclaimer.
//
//   * Redistribution's in binary form must reproduce the above copyright notice,
//     this list of conditions and the following disclaimer in the documentation
//     and/or other materials provided with the distribution.
//
//   * The name of the copyright holders may not be used to endorse or promote products
//     derived from this software without specific prior written permission.
//
//  This software is provided by the copyright holders and contributors "as is" and
//  any express or implied warranties, including, but not limited to, the implied
//  warranties of merchantability and fitness for a particular purpose are disclaimed.
//  In no event shall the Intel Corporation or contributors be liable for any direct,
//  indirect, incidental, special, exemplary, or consequential damages
//  (including, but not limited to, procurement of substitute goods or services;
//  loss of use, data, or profits; or business interruption) however caused
//  and on any theory of liability, whether in contract, strict liability,
//  or tort (including negligence or otherwise) arising in any way out of
//  the use of this software, even if advised of the possibility of such damage.

#ifndef LTS2_PROX_H
#define LTS2_PROX_H

#include <opencv2/core/core.hpp>

/**
 * The functions in this part are used to solve the so-called "prox" problems, i.e. 
 * proximal  operator computations. They return an updated point x such that:
 * x = prox_{\tau F}(y) = argmin \frac{\|x - y\|_2^2}{2 \tau} + F(x).
 */

namespace lts2 
{
    void ProxLinf(cv::Mat& X, float radius);
    
    void ProxL0(cv::Mat &X, int k);
    
    void ProxL2UnitBall(cv::Mat &X1, cv::Mat &X2);
    
    void ProxRangeConstraint(cv::Mat& X, float xmin, float xmax);
    void ProxMeanConstraint(cv::Mat& X, float mean);
    
    void ProxProjectOntoUnitSimplex(float* X, int n, int step=1);
    void ProxProjectOntoSimplexGen(float* X, float targetMean, int n, int step=1);
    
    void SoftThresholding(cv::Mat& X, float threshold);
    void SoftThresholding(cv::Mat &X1, cv::Mat &X2, float threshold);
}

#endif  // LTS2_PROX_H
