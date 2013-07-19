//  reconstruction.h
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

#ifndef LBD_RECONSTRUCTION_H
#define LBD_RECONSTRUCTION_H

#include <string>
#include <opencv2/core/core.hpp>
#include "ILBDOperator.hpp"

namespace lts2
{
  void RealReconstructionWithTVL1(cv::Mat &x, LBDOperator &LBD, cv::Mat const &binaryDescriptor,
                                  int iterations, float lambda, cv::Size patchSize,
                                  float patchMean=0.5);

  void RealReconstructionWithWL1(cv::Mat &x, LBDOperator &LBD, cv::Mat const &binaryDescriptor,
                                 int iterations, float lambda, cv::Size patchSize,
                                 float patchMean=0.5, std::string const &wavelet="haar");

  void BinaryReconstructionWithBIHT(cv::Mat &x, LBDOperator &LBD, cv::Mat const &binaryDescriptor,
                                    int iterations, float sparsityCoeff, cv::Size patchSize,
                                    float patchMean=0.5, int norm=cv::NORM_L1, std::string const &wavelet="haar");

  void RealReconstructionWithIHT(cv::Mat &X, LBDOperator &LBD, cv::Mat const &realDescriptor,
                                 int iterations, float sparsityCoeff, cv::Size patchSize,
                                 float patchMean=0.5, int norm=cv::NORM_L1, std::string const &wavelet="haar");

  void PerformTVL1OnImage(cv::Mat const &anImage, cv::Size patchSize, cv::Point patchOffset,
                          LBDOperator &LBD, cv::Mat &result, int iterations=200, float lambda=1e-1, float patchMean=0.5);
  void PerformTVL1OnImageFAST(cv::Mat const &anImage, cv::Size patchSize,
                              LBDOperator &LBD, cv::Mat &result, int iterations=200, float lambda=1e-1, float patchMean=0.5);

  void PerformBIHTOnImage(cv::Mat const &anImage, cv::Size patchSize, cv::Point patchOffset,
                          LBDOperator &LBD, cv::Mat &result, int iterations=1000, float sparsity=0.4, float patchMean=0.5);
  void PerformRIHTOnImage(cv::Mat const &anIMage, cv::Size patchSize, cv::Point patchOffset,
                          LBDOperator &LBD, cv::Mat &result, int iterations=1000, float sparsity=0.4, float patchMean=0.5);

  void PerformBIHTOnImageFAST(cv::Mat const &anImage, cv::Size patchSize,
                              LBDOperator &LBD, cv::Mat &result, int iterations=1000, float sparsity=0.4, float patchMean=0.5);
  void PerformRIHTOnImageFAST(cv::Mat const &anImage, cv::Size patchSize,
                              LBDOperator &LBD, cv::Mat &result, int iterations=1000, float sparsity=0.4, float patchMean=0.5);

  void PerformWL1OnImage(cv::Mat const &anImage, cv::Size patchSize, cv::Point patchOffset,
                         LBDOperator &LBD, cv::Mat &result, int iterations=200, float lambda=1e-1, float patchMean=0.5);
  void PerformWL1OnImageFAST(cv::Mat const &anImage, cv::Size patchSize,
                             LBDOperator &LBD, cv::Mat &result, int iterations=200, float lambda=1e-1, float patchMean=0.5);
}

#endif  // LBD_RECONSTRUCTION_H
