//  wavelets.h
//
//	Copyright (C) 2011-2012  Signal Processing Laboratory 2 (LTS2), EPFL,
//	Emmanuel d'Angelo (emmanuel.dangelo@epfl.ch),
//  and Jérôme Gilles (jegilles@math.ucla.edu)
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

#ifndef WAVELETS_H
#define WAVELETS_H

#include <opencv2/core/core.hpp>
#include <string>

namespace lts2 {
    
    /**
     * Method computing the wavelet coefficients for a given name of mthoer wavelet
     * @param name Name of the wavelet
     * @param P Receiver for the pointer to the wavelets coefficient values
     * @param Q Receiver for the pointer to the wavelets coefficient values
     * @param fsize Size of the allocated polynom
     */
    void WaveletCoef(const std::string& name, double **P, double **Q,int *fsize);
    
    void WaveletTransform2D(const cv::Mat& src, cv::Mat& dest, double *P, double *Q, 
                            int fsize, int *ww, int *wh, int stop_after=-1);
    
    void InverseWaveletTransform2D(const cv::Mat& src, cv::Mat& dest, 
                                   double *P, double *Q, int fsize, int *ww, int *wh);
    
    void ImageWaveletScale(const cv::Mat& src, cv::Mat& dest, 
                           double *P, double *Q, int fsize, int scale, int *rscale);
    
    void WaveletCoef_Biortho(const std::string& name,double **P,double **Q,int *fsizeP, int *fsizeQ);
    
    void CreateWaveletDisplay(const cv::Mat& waves, cv::Mat& display, int ww, int wh);
    
    void SoftThresholding(const cv::Mat& src, cv::Mat& thresholdedCoeffs, float threshold);
    void SoftThresholdingPreserveContinuous(const cv::Mat& src, cv::Mat& thresholdedCoeffs, float threshold);
    void KillDiagonalWaveletCoefficients(const cv::Mat& src, cv::Mat& cleaned, int ww, int wh);
    
    void HardThresholding(const cv::Mat& src, cv::Mat& thresholdedCoeffs, float threshold);
    
    /**
     * Perform wavelet transform by fast lifting transform, using symmetric boundary conditions.
     * Originally based on the matlab code by Gabriel Peyré.
     */
    void WaveletTransform(cv::Mat const &X, int Jmin, std::string const &wavelet="9-7", unsigned int options=0);
}

#endif  // WAVELETS_H
