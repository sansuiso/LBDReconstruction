//  derivatives.h
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

#ifndef LTS2_DERIVATIVES_H
#define LTS2_DERIVATIVES_H

#include <opencv2/core/core.hpp>

namespace lts2 {

#pragma mark - Horizontal gradient
    /**
     * Horizontal derivative with forward scheme
     * @param X Input image of type CV_32F
     * @param Dx Output image \nabla_x X
     * @see HorizontalGradientWithBackwardScheme HorizontalGradientWith5PointsScheme HorizontalGradientWithCenteredScheme
     */
    void HorizontalGradientWithForwardScheme(const cv::Mat& X, cv::Mat& Dx);

    /**
     * Horizontal derivative with backward scheme
     * @param X Input image of type CV_32F
     * @param Dx Output image \nabla_x X
     * @see HorizontalGradientWithForwardScheme HorizontalGradientWith5PointsScheme HorizontalGradientWithCenteredScheme
     */
    void HorizontalGradientWithBackwardScheme(const cv::Mat& X, cv::Mat& Dx);

#pragma mark - Vertical gradient
    /**
     * Vertical derivative with forward scheme
     * @param X Input image of type CV_32F
     * @param Dx Output image $\nabla_y X$
     * @see VerticalGradientWithBackwardScheme VerticalGradientWith5PointsScheme VerticalGradientWithCenteredScheme
     */
    void VerticalGradientWithForwardScheme(const cv::Mat& X, cv::Mat& Dx);
    
    /**
     * Vertical derivative with backward scheme
     * @param X Input image of type CV_32F
     * @param Dx Output image $\nabla_y X$
     * @see VerticalGradientWithForwardScheme VerticalGradientWith5PointsScheme VerticalGradientWithCenteredScheme
     */
    void VerticalGradientWithBackwardScheme(const cv::Mat& X, cv::Mat& Dx);
    
#pragma mark - Divergence (-adj(nabla))
    void DivergenceWithBackwardScheme(const cv::Mat& X1, const cv::Mat& X2, cv::Mat& divX);
    

#pragma mark - TV
    float NormTV(cv::Mat const &X);
    float NormTV(cv::Mat const &DX1, cv::Mat const &DX2);
    float NormTV(cv::Mat const &X, cv::Mat &DX1, cv::Mat &DX2);
}

#endif  // LTS2_DERIVATIVES_H
