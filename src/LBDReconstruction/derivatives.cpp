//  derivatives.cpp
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

#include "derivatives.h"

#include <iostream>
#include <opencv2/imgproc/imgproc.hpp>

#ifdef WITH_DISPATCH
#include <dispatch/dispatch.h>
#endif

using cv::Mat;

#pragma mark Divergence

/**
 * Compute the divergence of X = (X1, X2) using a backward scheme
 */
void lts2::DivergenceWithBackwardScheme(const Mat& X1, const Mat& X2, Mat& divX)
{	
    if (!X1.data || !X2.data)
        return;
    
	divX.create(X1.size(), CV_32F);
	divX.setTo(cv::Scalar::all(0));
    
	cv::Mat DX1;
    lts2::HorizontalGradientWithBackwardScheme(X1, DX1);
	
	cv::Mat DX2;
    lts2::VerticalGradientWithBackwardScheme(X2, DX2);
	
	divX = DX1 + DX2;
}

#pragma mark -
#pragma mark Horizontal gradient

void lts2::HorizontalGradientWithForwardScheme(const Mat& X, Mat& Dx)
{
    if (!X.data)
        return;
    
	Dx.create(X.size(), CV_32F);
	Dx.setTo(cv::Scalar::all(0));
    
    int valuesPerRow = (X.cols-1) * X.channels();
    
#ifdef WITH_DISPATCH
    __block cv::Mat blockDx = Dx;
    
    dispatch_queue_t queue = dispatch_get_global_queue(DISPATCH_QUEUE_PRIORITY_HIGH, 0);
    dispatch_apply(X.rows, queue,
                   ^(size_t i) {
                       float const* xj = X.ptr<float>(i);
                       float const* xjp1 = xj + X.channels();
                       
                       float* p_dx = blockDx.ptr<float>(i);
                       
                       for (int j = 0; j < valuesPerRow; ++j, ++xj, ++xjp1, ++p_dx)
                           *p_dx = (*xjp1 - *xj);
                   });
#else
    for (int i=0; i<X.rows; ++i) 
    {
        const float *xj = X.ptr<float>(i);
        const float *xjp1 = xj + X.channels();
        
        float *pdx = Dx.ptr<float>(i);
        
        for (int j=0; j < valuesPerRow; ++j, ++xj, ++xjp1, ++pdx)
            *pdx = (*xjp1 - *xj);
    }
#endif
}

void lts2::HorizontalGradientWithBackwardScheme(const Mat& X, Mat& Dx)
{
    if (!X.data)
        return;

    Dx.create(X.size(), CV_32F);
    Dx.setTo(cv::Scalar::all(0));
    
    int valuesPerRow = (X.cols-1) * X.channels();

#ifdef WITH_DISPATCH
    __block cv::Mat blockDx = Dx;
    
    dispatch_queue_t queue = dispatch_get_global_queue(DISPATCH_QUEUE_PRIORITY_HIGH, 0);
    dispatch_apply(X.rows, queue,
                   ^(size_t i) {
                       float const* xjm1 = X.ptr<float>(i);
                       float const* xj = xjm1 + X.channels();
                       
                       float* p_dx = blockDx.ptr<float>(i) + X.channels();
                       
                       for (int j = 0; j < valuesPerRow; ++j, ++xj, ++xjm1, ++p_dx)
                           *p_dx = (*xj - *xjm1);
                   });
#else
    for (int i=0; i<X.rows; ++i) 
    {
        const float *xjm1 = X.ptr<float>(i);
        const float *xj = xjm1 + X.channels();
        
        float *pdx = Dx.ptr<float>(i) + X.channels();
        
        for (int j = 0; j < valuesPerRow; ++j, ++xj, ++xjm1, ++pdx)
            *pdx = (*xj - *xjm1);
    }
#endif
}

#pragma mark - Vertical gradient

void lts2::VerticalGradientWithForwardScheme(const Mat& X, Mat& Dx)
{
    if (!X.data)
        return;
    
    Dx.create(X.size(), CV_32F);
    Dx.setTo(cv::Scalar::all(0));
    
    int valuesPerRow = X.cols * X.channels();
    
#ifdef WITH_DISPATCH
    __block cv::Mat blockDx = Dx;
    
    dispatch_queue_t queue = dispatch_get_global_queue(DISPATCH_QUEUE_PRIORITY_HIGH, 0);
    dispatch_apply(X.rows-1, queue,
                   ^(size_t i) {
                       // Current row
                       float const* xi = X.ptr<float>(i);
                       // Next row
                       float const* xip1 = X.ptr<float>(i+1);
                       
                       float* p_dx = blockDx.ptr<float>(i);
                       
                       for (int j = 0; j < valuesPerRow; ++j, ++xi, ++xip1, ++p_dx)
                           *p_dx = (*xip1 - *xi);
                   });
#else
    for (int i=0; i<X.rows-1; ++i) 
    {
        // Current row
        const float *xi = X.ptr<float>(i);
        // Next row
        const float *xip1 = X.ptr<float>(i+1);
        
        float *pdy = Dx.ptr<float>(i);
        
        for (int j = 0; j < valuesPerRow; ++j, ++xi, ++xip1, ++pdy)
            *pdy = (*xip1 - *xi);
    }
#endif
}

void lts2::VerticalGradientWithBackwardScheme(const Mat& X, Mat& Dx)
{
    if (!X.data)
        return;

    Dx.create(X.size(), CV_32F);
    Dx.setTo(cv::Scalar::all(0));
    
    int valuesPerRow = X.channels() * X.cols;
    
#ifdef WITH_DISPATCH
    __block cv::Mat blockDx = Dx;
    
    dispatch_queue_t queue = dispatch_get_global_queue(DISPATCH_QUEUE_PRIORITY_HIGH, 0);
    dispatch_apply(X.rows-1, queue,
                   ^(size_t i) {
                       // Current row
                       float const* xi = X.ptr<float>(i+1);
                       // Next row
                       float const* xim1 = X.ptr<float>(i);
                       
                       float* p_dx = blockDx.ptr<float>(i+1);
                       
                       for (int j = 0; j < valuesPerRow; ++j, ++xi, ++xim1, ++p_dx)
                           *p_dx = (*xi - *xim1);
                   });
#else
    for (int i = 1; i < X.rows; ++i) 
    {
        // Current row
        const float *xi = X.ptr<float>(i);
        // Previous row
        const float *xim1 = X.ptr<float>(i-1);
        
        float *pdy = Dx.ptr<float>(i);
        
        for (int j = 0; j < valuesPerRow; ++j, ++xi, ++xim1, ++pdy)
            *pdy = (*xi - *xim1);
    }
#endif
}

#pragma mark - TV

float lts2::NormTV(cv::Mat const &X)
{
    if (!X.data)
    {
        return 0.0;
    }
    
    cv::Mat DX1, DX2;
    
    lts2::HorizontalGradientWithForwardScheme(X, DX1);
    lts2::VerticalGradientWithForwardScheme(X, DX2);
    
    return lts2::NormTV(DX1, DX2);
}

float lts2::NormTV(cv::Mat const &DX1, cv::Mat const &DX2)
{
    if (!DX1.data || !DX2.data)
    {
        return 0.0;
    }
    
    cv::Mat magnitude;
    cv::magnitude(DX1, DX2, magnitude);
    
    return cv::norm(magnitude, cv::NORM_L1);
}

float lts2::NormTV(cv::Mat const &X, cv::Mat &DX1, cv::Mat &DX2)
{
    if (!X.data)
    {
        return 0.0;
    }
    
    lts2::HorizontalGradientWithForwardScheme(X, DX1);
    lts2::VerticalGradientWithForwardScheme(X, DX2);
    
    return lts2::NormTV(DX1, DX2);    
}
