//  reconstruction.cpp
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

#include <iostream>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/features2d/features2d.hpp>

#include "derivatives.h"
#include "prox.h"
#include "wavelets.h"
#include "SignOperator.hpp"
#include "reconstruction.h"

#ifdef WITH_DISPATCH
#include <dispatch/dispatch.h>
#endif

#ifndef LTS2_EPSILON
#define LTS2_EPSILON 1e-6
#endif

static
void finalize(cv::Mat &result, cv::Mat const &ocount)
{
#ifdef WITH_DISPATCH
  dispatch_queue_t queue = dispatch_get_global_queue(DISPATCH_QUEUE_PRIORITY_HIGH, 0);
  dispatch_apply(result.rows, queue,
                 ^(size_t y) {
                   float* p_res = result.ptr<float>(y);
                   float const* p_count = ocount.ptr<float>(y);
                   
                   for (int x = 0; x < result.cols; ++x, ++p_res, ++p_count)
                     if ( *p_count > 0.0 )
                       *p_res /= *p_count;
                 });
#else
  for (int y = 0; y < result.rows; ++y)
  {
    float* p_res = result.ptr<float>(y);
    float const* p_count = ocount.ptr<float>(y);

    for (int x = 0; x < result.cols; ++x, ++p_res, ++p_count)
      if ( *p_count > 0.0 )
        *p_res /= *p_count;
  }
#endif
}

static
void ExtractFastKeypoints(cv::Mat const &anImage, std::vector<cv::KeyPoint> &fastPoints)
{
  fastPoints.clear();
  cv::FastFeatureDetector FAST;

  cv::Mat anImage8;
  if (anImage.type() == CV_32F)
    anImage.convertTo(anImage8, CV_8U, 255.0);
  else
    anImage8 = anImage;

  cv::Mat srcImage;
  if (anImage8.channels() == 1)
    cv::cvtColor(anImage8, srcImage, CV_GRAY2BGR);
  else
    srcImage = anImage8;

  FAST.detect(srcImage, fastPoints);
  std::cerr << "Detected " << fastPoints.size() << " FAST\n";
}

static
void ExtractTheNegativePart(cv::Mat &src, cv::Mat &dest)
{
  CV_Assert(src.type() == CV_32F);
  dest.create(src.size(), src.type());

  int rows = src.rows;
  
  if (src.isContinuous() && dest.isContinuous())
  {
    src.cols *= rows;
    src.rows = 1;

    dest.cols *= rows;
    dest.rows = 1;
  }

#ifdef WITH_DISPATCH
  dispatch_queue_t queue = dispatch_get_global_queue(DISPATCH_QUEUE_PRIORITY_HIGH, 0);
  dispatch_apply(src.rows, queue,
                 ^(size_t y) {
                   float const *p_src = src.ptr<float>(y);
                   float *p_dest = dest.ptr<float>(y);

                   for (int x = 0; x < src.cols; ++x, ++p_src, ++p_dest)
                     if (*p_src < 0.0) *p_dest = -1.0;
                     else *p_dest = 0.0;
                 });
#else
  for (int y = 0; y < src.rows; ++y)
  {
    float const *p_src = src.ptr<float>(y);
    float *p_dest = dest.ptr<float>(y);

    for (int x = 0; x < src.cols; ++x, ++p_src, ++p_dest)
      if (*p_src < 0.0) *p_dest = -1.0;
      else *p_dest = 0.0;
  }

  if (src.isContinuous() && dest.isContinuous())
  {
    src.rows = rows;
    src.cols /= rows;

    dest.rows = rows;
    dest.cols /= rows;
  }
#endif
}

void lts2::RealReconstructionWithTVL1(cv::Mat &x, LBDOperator &LBD, cv::Mat const &binaryDescriptor,
                                     int iterations, float lambda, cv::Size patchSize, float patchMean)
{
  // Numerical parameters
  float L2 = 8.0f + LBD.norm() * LBD.norm();    // 8 is for TV
  float tau = 1.0f / std::sqrt(L2);   // L^2 * tau * sigma = 1
  float sigma = 1.0f / std::sqrt(L2); // L^2 * tau * sigma = 1

  // Auxiliary variables
  cv::Mat u_, Au, u_n, u_nm1, Du1, Du2;
  cv::Mat p1, p2, divP;       // Dual variables for TV
  cv::Mat q;                  // Dual variable for A

  // We observe g, hence g and Au have the same dimensions
  Au.create(binaryDescriptor.size(), CV_32F);
  Au.setTo(cv::Scalar(0));

  // The size of u may not be the same as the size of g (e.g. A is a projection)
  // so we guess the size of u either from a non-zero input, or from the application
  // of A^t.
  if (x.data)
    x.copyTo(u_n);
  else
  {
    LBD.ApplyConjugate(binaryDescriptor, u_n);
    u_n.setTo(cv::Scalar(0));
  }

  u_n.copyTo(u_);

  p1.create(u_.size(), CV_32FC1);
  p1.setTo(cv::Scalar(0));
  p2.create(u_.size(), CV_32FC1);
  p2.setTo(cv::Scalar(0));

  q.create(binaryDescriptor.size(), CV_32F);
  q.setTo(cv::Scalar(0));

  // Main loop
  for (int i = 0; i < iterations; ++i)
  {
    //-------------------------------
    // Solve the dual (F^\star, K)
    //-------------------------------

    // K = (\nabla, A)
    // Apply K to U_, add to the dual variable

    // First part: TV
    lts2::HorizontalGradientWithForwardScheme(u_, Du1);
    Du1 *= sigma;
    p1 += Du1;

    lts2::VerticalGradientWithForwardScheme(u_, Du2);
    Du2 *= sigma;
    p2 += Du2;

    lts2::ProxL2UnitBall(p1, p2);

    // Second part: Au
    LBD.Apply(u_, Au);
    Au -= binaryDescriptor;
    Au *= sigma;
    q += Au;
    lts2::ProxRangeConstraint(q, -lambda, lambda);

    // Compute the prox

    //-------------------------------
    // Solve the primal (G, K^\star)
    //-------------------------------

    // Save the current estimate
    u_n.copyTo(u_nm1);

    // Apply K^\star to the dual variable
    // TV
    lts2::DivergenceWithBackwardScheme(p1, p2, divP);
    divP *= tau;

    // Operator A
    LBD.ApplyConjugate(q, Au);
    Au *= tau;

    // Solve the prox                                                                                                                                                   
    u_n -= Au;
    u_n += divP;

    // Enforce the range of the data [0, 1] and the mean
    if (!u_n.isContinuous())
      std::cerr << "OOOOOOPS, things can go wild..." << std::endl;

    lts2::ProxMeanConstraint(u_n, patchMean);
    lts2::ProxRangeConstraint(u_n, 0.0, 1.0);

    //-------------------------------
    // Next iteration
    //-------------------------------
    u_ = 2.0f * u_n - u_nm1;
  }

  u_n.copyTo(x);
}

void lts2::RealReconstructionWithWL1(cv::Mat &x, LBDOperator &LBD, cv::Mat const &binaryDescriptor,
                                     int iterations, float lambda, cv::Size patchSize, float patchMean, std::string const &wavelet)
{
  // Numerical parameters (theta = 1 wrt Chambolle-Pock description)
  float L2 = 1.0 + LBD.norm() * LBD.norm();   // 1 is for W
  float tau = 1.0 / std::sqrt(L2);     // L^2 * tau * sigma = 1
  float sigma = 1.0 / std::sqrt(L2);   // L^2 * tau * sigma = 1

  // Auxiliary variables
  // Since x = (y,z) s.t. y = z, we do not explicitely allocate variables for y,z.
  // The y,z variabels are actually only used in the primal part to hold temp values.
  cv::Mat x_prev, x_bar;    // Auxiliary points (for algorithm speedup)
  cv::Mat y, Ly, r;            // y = LBD(x), r = dual(y)
  cv::Mat z, Wz, s;            // z = W(x), s = dual(z)

  // Wavelet basis
  double *P = NULL, *Q = NULL;
  int fsize = 0, ww, wh;
  lts2::WaveletCoef(wavelet, &P, &Q, &fsize);

  /////////////////////////////////////////////////
  //                                             //
  // Allocate memory with the correct dimensions //
  //                                             //
  /////////////////////////////////////////////////

  // The size of x may not be the same as the size of the observation (e.g.the operator is a projection)
  // so we guess the size of u either from a non-zero input, or from the application
  // of operator^T.
  LBD.ApplyConjugate(binaryDescriptor, x);
  x.setTo(cv::Scalar::all(0.0));

  x.copyTo(x_bar);

  x.copyTo(Wz);
  Wz.copyTo(s);

  x.copyTo(y);

  Ly.create(binaryDescriptor.size(), CV_32F);
  Ly.setTo(cv::Scalar(0));
  Ly.copyTo(r);
  
  // Main loop
  for (int i = 0; i < iterations; ++i)
  {
    ////////////////////////
    //                    //
    //   Solve the dual   //
    //                    //
    ////////////////////////

    // First part: LBD
    LBD.Apply(x_bar, Ly);
    Ly -= binaryDescriptor;
    cv::addWeighted(Ly, sigma, r, 1.0, 0.0, r);
    lts2::ProxRangeConstraint(r, -lambda, lambda);

    // Second part: sparsity
    lts2::WaveletTransform2D(x_bar, Wz, P, Q, fsize, &ww, &wh);
    cv::addWeighted(Wz, sigma, s, 1.0, 0.0, s);
    lts2::ProxLinf(s, 1.0);

    //////////////////////////
    //                      //
    //   Solve the primal   //
    //                      //
    //////////////////////////

    // Save the current estimate
    x.copyTo(x_prev);

    // Apply K^\star to the dual variables

    // First part: LBD
    LBD.ApplyConjugate(r, y);
    cv::addWeighted(x, 1.0, y, -0.5*tau, 0.0, x);
    
    // Second part: sparsity
    lts2::InverseWaveletTransform2D(s, z, P, Q, fsize, &ww, &wh);
    cv::addWeighted(x, 1.0, z, -0.5*tau, 0.0, x);

    // Solve the proximal mappings
    // Enforce the range of the data [0,1] and the mean
    // The projection onto the simplex may be buggy if the data is not continuous.
    if (!x.isContinuous())
      std::cerr << "OOOOOOPS, things can go wild..." << std::endl;
    lts2::ProxMeanConstraint(x, patchMean);
    lts2::ProxRangeConstraint(x, 0.0, 1.0);

    //-------------------------------
    // Next iteration
    //-------------------------------
    x_bar = 2.0*x - x_prev;
  }

  // Release memory
  delete P;
  delete Q;
}

void lts2::BinaryReconstructionWithBIHT(cv::Mat &X, LBDOperator &LBD, cv::Mat const &binaryDescriptor,
                                        int iterations, float sparsityCoeff, cv::Size patchSize, float patchMean,
                                        int norm, std::string const &wavelet)
{
  CV_Assert(norm == CV_L1 || norm == CV_L2);

  int const M = MAX(binaryDescriptor.rows, binaryDescriptor.cols);
  float const tau = 1.0 / (float)M;

  sparsityCoeff = MAX(MIN(sparsityCoeff, 1.0), 0.0);
  int const K = (int)rintf( sparsityCoeff*patchSize.area() );

  X.create(patchSize, CV_32FC1);
  X.setTo(cv::Scalar(0.0));

  cv::Mat X_im;
  cv::Mat Ax;
  cv::Mat nablaJ;

  // Wavelet basis
  double *P = NULL, *Q = NULL;
  int fsize = 0, ww, wh;
  lts2::WaveletCoef(wavelet, &P, &Q, &fsize);

  X_im.create(patchSize, CV_32F);
  X_im.setTo(cv::Scalar(patchMean));
  lts2::WaveletTransform2D(X_im, X, P, Q, fsize, &ww, &wh);
  cv::Mat C0 = X(cv::Rect(0,0,ww,wh)).clone();
  lts2::InverseWaveletTransform2D(X, X_im, P, Q, fsize, &ww, &wh);

  // Operators
  SignOperator A;

  // Special init if norm = L2
  X.create(patchSize, CV_32FC1);
  X.setTo(cv::Scalar(0.0));
  if (norm == CV_L2)
  {
    LBD.ApplyConjugate(binaryDescriptor, X_im);
    lts2::WaveletTransform2D(X_im, X, P, Q, fsize, &ww, &wh);
    float norm_x = cv::norm(X);
    if (norm_x > 0.0) X /= norm_x;
  }

  bool hasConverged = false;
  int iter = 0;

  while (!hasConverged)
  {
    // Back from the wavelet world
    lts2::InverseWaveletTransform2D(X, X_im, P, Q, fsize, &ww, &wh);
    LBD.Apply(X_im, Ax);

    // Step 1a : gradient
    if (norm == CV_L1)
    {
      Ax -= binaryDescriptor;

      LBD.ApplyConjugate(Ax, X_im);
      lts2::WaveletTransform2D(X_im, nablaJ, P, Q, fsize, &ww, &wh);
      nablaJ *= (0.5*tau);
    }
    else
    {
      cv::multiply(Ax, binaryDescriptor, Ax);
      ExtractTheNegativePart(Ax, Ax);

      cv::multiply(Ax, binaryDescriptor, Ax);
      LBD.ApplyConjugate(Ax, X_im);
      lts2::WaveletTransform2D(X_im, nablaJ, P, Q, fsize, &ww, &wh);

      nablaJ *= tau;
    }

    // Step 1b : update estimate
    X -= nablaJ;

    // Step 2 : enforce sparsity
    lts2::ProxL0(X, K);
    lts2::InverseWaveletTransform2D(X, X_im, P, Q, fsize, &ww, &wh);
    lts2::ProxRangeConstraint(X_im, 0.0, 1.0);
    lts2::ProxMeanConstraint(X_im, patchMean);

    // Termination criterion
    hasConverged = (++iter >= iterations);  // Iterations count limit
  }

#ifdef DEBUG
  std::cout << "Stopped after " << iter << " iterations " << " (max allowed was " << iterations << ")" << std::endl;
#endif

  // Normalize by projecting to the sphere
  float norm_x = cv::norm(X);
//  if (norm_x < LTS2_EPSILON)
//    norm_x = 1.0
//  X /= norm_x;

  lts2::InverseWaveletTransform2D(X, X_im, P, Q, fsize, &ww, &wh);
  cv::normalize(X_im, X, 0.0, 1.0, cv::NORM_MINMAX);

  // Cleanup
  delete[] P;
  delete[] Q;
}

void lts2::PerformTVL1OnImage(cv::Mat const &anImage, cv::Size patchSize, cv::Point patchOffset,
                             LBDOperator &LBD, cv::Mat &result, int iterations, float lambda, float patchMean)
{
  CV_Assert(anImage.type() == CV_32F && anImage.channels() == 1);

  // Result image
  result.create(anImage.size(), CV_32F);
  result.setTo(cv::Scalar(0));

  cv::Mat ocount(anImage.size(), CV_32FC1, cv::Scalar(0));
  cv::Mat patchOnes = cv::Mat::ones(patchSize, CV_32FC1);

  // Loop over the patches
  cv::Mat inputPatch, freak;
  cv::Mat resultPatch;
  patchOffset.x = MAX(1, patchOffset.x);
  patchOffset.y = MAX(1, patchOffset.y);

  for (int y = 0; y < anImage.rows; y += patchOffset.y)
    for (int x = 0; x < anImage.cols; x += patchOffset.x)
    {
      cv::Rect ROI(x, y, patchSize.width, patchSize.height);
      if ( y+patchSize.height > anImage.rows ||
           x+patchSize.width > anImage.cols )
        continue;

      anImage(ROI).copyTo(inputPatch);
      float pmean = patchMean;
      if (pmean < 0.0)
      {
        cv::Scalar mean = cv::mean(inputPatch);
        pmean = (float)mean[0];
      }

      LBD.Apply(inputPatch, freak);
      lts2::RealReconstructionWithTVL1(resultPatch, LBD, freak, iterations,
                                      lambda, patchSize, pmean);

      cv::Mat target = result(ROI);
      target += resultPatch;

      target = ocount(ROI);
      target += patchOnes;
    }

  finalize(result, ocount);
}

void lts2::PerformTVL1OnImageFAST(cv::Mat const &anImage, cv::Size patchSize,
                                 LBDOperator &LBD, cv::Mat &result, int iterations, float lambda, float patchMean)
{
  CV_Assert(anImage.type() == CV_32F && anImage.channels() == 1);

  // Extract image FAST points
  std::vector<cv::KeyPoint> fastPoints;
  ExtractFastKeypoints(anImage, fastPoints);

  // Result image
  result.create(anImage.size(), CV_32F);
  result.setTo(cv::Scalar(0));

  cv::Mat ocount(anImage.size(), CV_32FC1, cv::Scalar(0));
  cv::Mat patchOnes = cv::Mat::ones(patchSize, CV_32FC1);

  // Loop over the patches
  cv::Mat inputPatch, freak;
  cv::Mat resultPatch;
  int const pradius_x = patchSize.width / 2;
  int const pradius_y = patchSize.height / 2;

  std::vector<cv::KeyPoint>::const_iterator p_fast;

  for (p_fast = fastPoints.begin(); p_fast != fastPoints.end(); ++p_fast)
  {
    int x = (int)rintf(p_fast->pt.x) - pradius_x;
    int y = (int)rintf(p_fast->pt.y) - pradius_y;

    cv::Rect ROI(x, y, patchSize.width, patchSize.height);
    if ( y < 0 || y+patchSize.height > anImage.rows ||
         x < 0 || x+patchSize.width > anImage.cols )
      continue;

    anImage(ROI).copyTo(inputPatch);
    float pmean = patchMean;
    if (pmean < 0.0)
    {
      cv::Scalar mean = cv::mean(inputPatch);
      pmean = (float)mean[0];
    }

    LBD.Apply(inputPatch, freak);
    lts2::RealReconstructionWithTVL1(resultPatch, LBD, freak, iterations,
                                    lambda, patchSize, pmean);

    cv::Mat target = result(ROI);
    target += resultPatch;

    target = ocount(ROI);
    target += patchOnes;
  }

  finalize(result, ocount);
}

void lts2::PerformWL1OnImage(cv::Mat const &anImage, cv::Size patchSize, cv::Point patchOffset,
                             LBDOperator &LBD, cv::Mat &result, int iterations, float lambda, float patchMean)
{
  CV_Assert(anImage.type() == CV_32F && anImage.channels() == 1);

  // Result image
  result.create(anImage.size(), CV_32F);
  result.setTo(cv::Scalar(0));

  cv::Mat ocount(anImage.size(), CV_32FC1, cv::Scalar(0));
  cv::Mat patchOnes = cv::Mat::ones(patchSize, CV_32FC1);

  // Loop over the patches
  cv::Mat inputPatch, freak;
  cv::Mat resultPatch;
  patchOffset.x = MAX(1, patchOffset.x);
  patchOffset.y = MAX(1, patchOffset.y);

  for (int y = 0; y < anImage.rows; y += patchOffset.y)
    for (int x = 0; x < anImage.cols; x += patchOffset.x)
    {
      cv::Rect ROI(x, y, patchSize.width, patchSize.height);
      if ( y+patchSize.height > anImage.rows ||
           x+patchSize.width > anImage.cols )
        continue;

      anImage(ROI).copyTo(inputPatch);
      float pmean = patchMean;
      if (pmean < 0.0)
      {
        cv::Scalar mean = cv::mean(inputPatch);
        pmean = (float)mean[0];
      }

      LBD.Apply(inputPatch, freak);
      lts2::RealReconstructionWithWL1(resultPatch, LBD, freak, iterations,
                                      lambda, patchSize, pmean);

      cv::Mat target = result(ROI);
      target += resultPatch;

      target = ocount(ROI);
      target += patchOnes;
    }

  finalize(result, ocount);
}

void lts2::PerformWL1OnImageFAST(cv::Mat const &anImage, cv::Size patchSize,
                                 LBDOperator &LBD, cv::Mat &result, int iterations, float lambda, float patchMean)
{
  CV_Assert(anImage.type() == CV_32F && anImage.channels() == 1);

  // Extract image FAST points
  std::vector<cv::KeyPoint> fastPoints;
  ExtractFastKeypoints(anImage, fastPoints);

  // Result image
  result.create(anImage.size(), CV_32F);
  result.setTo(cv::Scalar(0));

  cv::Mat ocount(anImage.size(), CV_32FC1, cv::Scalar(0));
  cv::Mat patchOnes = cv::Mat::ones(patchSize, CV_32FC1);

  // Loop over the patches
  cv::Mat inputPatch, freak;
  cv::Mat resultPatch;
  int const pradius_x = patchSize.width / 2;
  int const pradius_y = patchSize.height / 2;

  std::vector<cv::KeyPoint>::const_iterator p_fast;

  for (p_fast = fastPoints.begin(); p_fast != fastPoints.end(); ++p_fast)
  {
    int x = (int)rintf(p_fast->pt.x) - pradius_x;
    int y = (int)rintf(p_fast->pt.y) - pradius_y;

    cv::Rect ROI(x, y, patchSize.width, patchSize.height);
    if ( y < 0 || y+patchSize.height > anImage.rows ||
         x < 0 || x+patchSize.width > anImage.cols )
      continue;

    anImage(ROI).copyTo(inputPatch);
    float pmean = patchMean;
    if (pmean < 0.0)
    {
      cv::Scalar mean = cv::mean(inputPatch);
      pmean = (float)mean[0];
    }

    LBD.Apply(inputPatch, freak);
    lts2::RealReconstructionWithWL1(resultPatch, LBD, freak, iterations,
                                    lambda, patchSize, pmean);

    cv::Mat target = result(ROI);
    target += resultPatch;

    target = ocount(ROI);
    target += patchOnes;
  }

  finalize(result, ocount);
}

void lts2::PerformBIHTOnImage(cv::Mat const &anImage, cv::Size patchSize, cv::Point patchOffset,
                             LBDOperator &LBD, cv::Mat &result, int iterations, float sparsity, float patchMean)
{
  CV_Assert(anImage.type() == CV_32F && anImage.channels() == 1);

  // Result image  
  result.create(anImage.size(), CV_32F);
  result.setTo(cv::Scalar(0));

  cv::Mat ocount(anImage.size(), CV_32FC1, cv::Scalar(0));
  cv::Mat patchOnes = cv::Mat::ones(patchSize, CV_32FC1);

  // Binarization operator
  SignOperator A;

  // Loop over the patches
  cv::Mat inputPatch, freak;
  cv::Mat resultPatch;
  patchOffset.x = MAX(1, patchOffset.x);
  patchOffset.y = MAX(1, patchOffset.y);

  for (int y = 0; y < anImage.rows; y += patchOffset.y)
    for (int x = 0; x < anImage.cols; x += patchOffset.x)
    {
      cv::Rect ROI(x, y, patchSize.width, patchSize.height);
      if ( y+patchSize.height > anImage.rows ||
           x+patchSize.width > anImage.cols )
        continue;

      anImage(ROI).copyTo(inputPatch);
      float pmean = patchMean;
      if (pmean < 0.0)
      {
        cv::Scalar mean = cv::mean(inputPatch);
        pmean = (float)mean[0];
      }

      LBD.Apply(inputPatch, freak);
      A.Apply(freak, freak);

      lts2::BinaryReconstructionWithBIHT(resultPatch, LBD, freak,
                                         iterations, sparsity, patchSize, pmean);

      cv::Mat target = result(ROI);
      target += resultPatch;

      target = ocount(ROI);
      target += patchOnes;
    }

  finalize(result, ocount);
}

void lts2::PerformBIHTOnImageFAST(cv::Mat const &anImage, cv::Size patchSize,
                                 LBDOperator &LBD, cv::Mat &result, int iterations, float sparsity, float patchMean)
{
  CV_Assert(anImage.type() == CV_32F && anImage.channels() == 1);

  // Extract image FAST points
  std::vector<cv::KeyPoint> fastPoints;
  ExtractFastKeypoints(anImage, fastPoints);

  // Result image
  result.create(anImage.size(), CV_32F);
  result.setTo(cv::Scalar(0));

  cv::Mat ocount(anImage.size(), CV_32FC1, cv::Scalar(0));
  cv::Mat patchOnes = cv::Mat::ones(patchSize, CV_32FC1);

  // Binarization operator
  SignOperator A;

  // Loop over the patches
  cv::Mat inputPatch, freak;
  cv::Mat resultPatch;
  int const pradius_x = patchSize.width / 2;
  int const pradius_y = patchSize.height / 2;
  
  std::vector<cv::KeyPoint>::const_iterator p_fast;
  
  for (p_fast = fastPoints.begin(); p_fast != fastPoints.end(); ++p_fast)
  {
    int x = (int)rintf(p_fast->pt.x) - pradius_x;
    int y = (int)rintf(p_fast->pt.y) - pradius_y;
    
    cv::Rect ROI(x, y, patchSize.width, patchSize.height);
    if ( y < 0 || y+patchSize.height > anImage.rows ||
         x < 0 || x+patchSize.width > anImage.cols )
      continue;
    
    anImage(ROI).copyTo(inputPatch);
    float pmean = patchMean;
    if (pmean < 0.0)
    {
      cv::Scalar mean = cv::mean(inputPatch);
      pmean = (float)mean[0];
    }
    
    LBD.Apply(inputPatch, freak);
    A.Apply(freak, freak);
    
    lts2::BinaryReconstructionWithBIHT(resultPatch, LBD, freak,
                                       iterations, sparsity, patchSize, pmean);
    
    cv::Mat target = result(ROI);
    target += resultPatch;
    
    target = ocount(ROI);
    target += patchOnes;
  }
  
  finalize(result, ocount);
}

void lts2::RealReconstructionWithIHT(cv::Mat &X, LBDOperator &LBD, cv::Mat const &realDescriptor,
                                     int iterations, float sparsityCoeff, cv::Size patchSize, float patchMean,
                                     int norm, std::string const &wavelet)
{
  CV_Assert(norm == CV_L1 || norm == CV_L2);
  CV_Assert(realDescriptor.type() == CV_32F);

  int const M = MAX(realDescriptor.rows, realDescriptor.cols);
  float const tau = (norm == CV_L2 ? 1.0/(float)M : 0.5/(float)M);

  sparsityCoeff = MAX(MIN(sparsityCoeff, 1.0), 0.0);
  int const K = (int)rintf( sparsityCoeff*patchSize.area() );

  X.create(patchSize, CV_32FC1);
  X.setTo(cv::Scalar(0.0));

  cv::Mat X_im;
  cv::Mat Ax;
  cv::Mat nablaJ;

  // Wavelet basis
  double *P = NULL, *Q = NULL;
  int fsize = 0, ww, wh;
  lts2::WaveletCoef(wavelet, &P, &Q, &fsize);

  X_im.create(patchSize, CV_32F);
  X_im.setTo(cv::Scalar(patchMean));
  lts2::WaveletTransform2D(X_im, X, P, Q, fsize, &ww, &wh);
  cv::Mat C0 = X(cv::Rect(0,0,ww,wh)).clone();
  lts2::InverseWaveletTransform2D(X, X_im, P, Q, fsize, &ww, &wh);

  // Special init if norm = L2
  X.create(patchSize, CV_32FC1);
  X.setTo(cv::Scalar(0.0));
  if (norm == CV_L2)
  {
    LBD.ApplyConjugate(realDescriptor, X_im);
    lts2::WaveletTransform2D(X_im, X, P, Q, fsize, &ww, &wh);
    float norm_x = cv::norm(X);
    if (norm_x > 0.0) X /= norm_x;
  }

  bool hasConverged = false;
  int iter = 0;

  while (!hasConverged)
  {
    // Back from the wavelet world
    lts2::InverseWaveletTransform2D(X, X_im, P, Q, fsize, &ww, &wh);
    LBD.Apply(X_im, Ax);
    Ax -= realDescriptor;

    // Step 1a : gradient
    if (norm == CV_L1)
    {
      float norm_error = cv::norm(Ax);
      norm_error = (norm_error > LTS2_EPSILON ? norm_error : 1.0);
      Ax /= norm_error;
    }

    LBD.ApplyConjugate(Ax, X_im);
    lts2::WaveletTransform2D(X_im, nablaJ, P, Q, fsize, &ww, &wh);
    nablaJ *= tau;

    X -= nablaJ;

    lts2::ProxL0(X, K);
    
    lts2::InverseWaveletTransform2D(X, X_im, P, Q, fsize, &ww, &wh);
    lts2::ProxRangeConstraint(X_im, 0.0, 1.0);
    lts2::ProxMeanConstraint(X_im, patchMean);
    lts2::WaveletTransform2D(X_im, X, P, Q, fsize, &ww, &wh);

    // Termination criterion
    hasConverged = (++iter >= iterations);  // Iterations count limit
    // TODO : test consistancy here (hasConverged |= ...)
  }

#ifdef DEBUG
  std::cout << "Stopped after " << iter << " iterations " << " (max allowed was " << iterations << ")" << std::endl;
#endif

  // Normalize by projecting to the sphere
  float norm_x = cv::norm(X);
  if (norm_x < LTS2_EPSILON)
    norm_x = 1.0;
  X /= norm_x;

  lts2::InverseWaveletTransform2D(X, X_im, P, Q, fsize, &ww, &wh);
  cv::normalize(X_im, X, 0.0, 1.0, cv::NORM_MINMAX);

  // Cleanup
  delete[] P;
  delete[] Q;
}

void lts2::PerformRIHTOnImage(cv::Mat const &anImage, cv::Size patchSize, cv::Point patchOffset,
                             LBDOperator &LBD, cv::Mat &result, int iterations, float sparsity, float patchMean)
{
  CV_Assert(anImage.type() == CV_32F && anImage.channels() == 1);

  // Result image
  result.create(anImage.size(), CV_32F);
  result.setTo(cv::Scalar(0));

  cv::Mat ocount(anImage.size(), CV_32FC1, cv::Scalar(0));
  cv::Mat patchOnes = cv::Mat::ones(patchSize, CV_32FC1);


  // Loop over the patches
  cv::Mat inputPatch, freak;
  cv::Mat resultPatch;
  patchOffset.x = MAX(1, patchOffset.x);
  patchOffset.y = MAX(1, patchOffset.y);

  for (int y = 0; y < anImage.rows; y += patchOffset.y)
    for (int x = 0; x < anImage.cols; x += patchOffset.x)
    {
      cv::Rect ROI(x, y, patchSize.width, patchSize.height);
      if ( y+patchSize.height > anImage.rows ||
           x+patchSize.width > anImage.cols )
        continue;

      anImage(ROI).copyTo(inputPatch);
      float pmean = patchMean;
      if (pmean < 0.0)
      {
        cv::Scalar mean = cv::mean(inputPatch);
        pmean = (float)mean[0];
      }

      LBD.Apply(inputPatch, freak);

      lts2::RealReconstructionWithIHT(resultPatch, LBD, freak,
                                      iterations, sparsity, patchSize, pmean);

      cv::Mat target = result(ROI);
      target += resultPatch;

      target = ocount(ROI);
      target += patchOnes;
    }

  finalize(result, ocount);
}

void lts2::PerformRIHTOnImageFAST(cv::Mat const &anImage, cv::Size patchSize,
                                 LBDOperator &LBD, cv::Mat &result, int iterations, float sparsity, float patchMean)
{
  CV_Assert(anImage.type() == CV_32F && anImage.channels() == 1);
  
  // Extract FAST keypoints
  std::vector<cv::KeyPoint> fastPoints;
  ExtractFastKeypoints(anImage, fastPoints);
  
  // Result image
  result.create(anImage.size(), CV_32F);
  result.setTo(cv::Scalar(0));
  
  cv::Mat ocount(anImage.size(), CV_32FC1, cv::Scalar(0));
  cv::Mat patchOnes = cv::Mat::ones(patchSize, CV_32FC1);
  
  // Loop over the patches
  cv::Mat inputPatch, freak;
  cv::Mat resultPatch;
  int const pradius_x = patchSize.width / 2;
  int const pradius_y = patchSize.height / 2;
  
  std::vector<cv::KeyPoint>::const_iterator p_fast;

  for (p_fast = fastPoints.begin(); p_fast != fastPoints.end(); ++p_fast)
  {
    int x = (int)rintf(p_fast->pt.x) - pradius_x;
    int y = (int)rintf(p_fast->pt.y) - pradius_y;
    
    cv::Rect ROI(x, y, patchSize.width, patchSize.height);
    if ( y < 0 || y+patchSize.height > anImage.rows ||
         x < 0 || x+patchSize.width > anImage.cols )
      continue;

    anImage(ROI).copyTo(inputPatch);
    float pmean = patchMean;
    if (pmean < 0.0)
    {
      cv::Scalar mean = cv::mean(inputPatch);
      pmean = (float)mean[0];
    }

    LBD.Apply(inputPatch, freak);

    lts2::RealReconstructionWithIHT(resultPatch, LBD, freak,
                                    iterations, sparsity, patchSize, pmean);

    cv::Mat target = result(ROI);
    target += resultPatch;

    target = ocount(ROI);
    target += patchOnes;
  }
  
  finalize(result, ocount);
}
