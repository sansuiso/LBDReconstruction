//  prox.cpp
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

#include "prox.h"

#include <iostream>
#include <cmath>

// For Grand Central Dispatch
#ifdef WITH_DISPATCH
#include <dispatch/dispatch.h>
#endif

#ifdef DEBUG
#undef WITH_DISPATCH
#endif

void lts2::ProxLinf(cv::Mat& X, float radius)
{
  CV_Assert(X.data && X.depth() == CV_32F);

#ifdef WITH_DISPATCH
  // Get a queue
  dispatch_queue_t queue = dispatch_get_global_queue(DISPATCH_QUEUE_PRIORITY_HIGH, 0);

  // Parallel loop over the rows
  float* p_imageData = X.ptr<float>(0);

  dispatch_apply(X.rows, queue,
                 ^(size_t y) {
                   float *p_x = X.ptr<float>(y);
                   
                   for (int x = 0; x < X.channels()*X.cols; ++x, ++p_x) {
                     if (std::fabs(*p_x) > radius) {
                       if (*p_x > 0.0)
                         *p_x = radius;
                       else
                         *p_x = -radius;
                     }
                   }
                 });    
#else
  for (int y = 0; y < X.rows; ++y)
  {
    float* p_x = X.ptr<float>(y);
    
    for (int x = 0; x < X.cols*X.channels(); ++x, ++p_x)
    {
      float abs_x = std::fabs(*p_x);
      if (abs_x > radius) {
        if (*p_x > 0.0) 
          *p_x = radius;
        else
        *p_x = -radius;
      }
    }
  }
#endif
}

void lts2::ProxL2UnitBall(cv::Mat& X1, cv::Mat& X2)
{
  for (int y = 0; y < X1.rows; ++y)
  {
    float *p1 = X1.ptr<float>(y);
    float *p2 = X2.ptr<float>(y);

    for (int x = 0; x < X1.cols; ++x, ++p1, ++p2)
    {
      float pnorm = MAX(1.0f, sqrtf((*p1)*(*p1) + (*p2)*(*p2)));
      *p1 /= pnorm;
      *p2 /= pnorm;
    }
  }
}

void lts2::ProxRangeConstraint(cv::Mat &X, float xmin, float xmax)
{
#ifdef WITH_DISPATCH
  dispatch_queue_t queue = dispatch_get_global_queue(DISPATCH_QUEUE_PRIORITY_HIGH, 0);
    
  // Create a local variable, which can be modified by the block.
  // Due to OpenCV memory management, we only create a header, no data is copied!
  __block cv::Mat localX(X);
    
  // Apply processing to each row
  dispatch_apply(X.rows, queue, ^(size_t y) {
      float *p_x = localX.ptr<float>(y);
      for (int x = 0; x < X.cols; ++x)
        *p_x++ = MIN(MAX(*p_x, xmin), xmax);
    });
#else
  for (int y = 0; y < X.rows; ++y)
  {
    float *p_x = X.ptr<float>(y);
        
    for (int x = 0; x < X.cols; ++x, ++p_x)
      *p_x = MIN(MAX(*p_x, xmin), xmax);
  }
#endif
}

void lts2::ProxL0(cv::Mat &X, int k)
{
  cv::Mat rowVector = X.reshape(1, X.rows*X.cols);

  cv::Mat sortedRow;
  cv::sortIdx(rowVector, sortedRow, cv::SORT_EVERY_ROW + cv::SORT_DESCENDING);

  float *p_row = rowVector.ptr<float>(0);
  int *p_idx = sortedRow.ptr<int>(0) + k;
  for (int i = k; i < rowVector.cols; ++i)
  {
    int col = *p_idx++;
    *(p_row+col) = 0.0;
  }
    
  X = rowVector.reshape(1, X.rows);
}

void lts2::ProxMeanConstraint(cv::Mat& X, float mean)
{
  cv::Scalar input_mean = cv::mean(X);
  float x_mean = float(input_mean.val[0]);
    
  for (int y = 0; y < X.rows; ++y)
  {
    float* p_x = X.ptr<float>(y);
        
    for (int x = 0; x < X.cols; ++x)
      *p_x++ = *p_x - x_mean + mean;
  }
}

void lts2::ProxProjectOntoUnitSimplex(float* X, int n, int step)
{
  unsigned char* In_complement = new unsigned char[n];
  for (int i = 0; i < n; ++i)
    In_complement[i] = 255;
    
  float* Xtilda = new float[n];
        
  // Iterate
  bool success = false;
  int iterationCount = 0;
  while ( !success && iterationCount < n ) 
  {
    success = true;
        
    // Compute the mean used to project onto V_I
    float mu = 0.0;
    int nonZero = 0;
    for (int i = 0; i < n; ++i)
      if ( In_complement[i] > 0 )
      {
        mu += X[i*step];
        ++nonZero;
      }
    mu -= 1.0;
    mu /= nonZero;
    
    // Compute projections onto V_I and X_I together
    // Note that we can test at the same time the convergence:
    // we have converged when all the components of X tilda are positive.
    for (int i = 0; i < n; ++i)
    {
      // Projection onto V_I : enforce the summation to 1
      if ( In_complement[i] > 0 )
        Xtilda[i] = X[i*step] - mu;
      else
        Xtilda[i] = 0.0;
            
      // Projection onto X_I : enforce the positivity
      X[i*step] = MAX(Xtilda[i], 0.0);
            
      // Convergence check
      if ( Xtilda[i] < 0.0 )
      {
        In_complement[i] = 0;
        success = false;
      }
    }
        
    ++iterationCount;
  }

    
  // Release memory
  delete []In_complement;
  delete []Xtilda;
}

void lts2::ProxProjectOntoSimplexGen(float* X, float targetMean, int n, int step)
{
  targetMean = MAX(0.0, targetMean);
    
  unsigned char* In_complement = new unsigned char[n];
  for (int i = 0; i < n; ++i)
    In_complement[i] = 255;
    
  float* Xtilda = new float[n];
    
  // Iterate
  bool success = false;
  int iterationCount = 0;
  while ( !success && iterationCount < n ) 
  {
    success = true;
        
    // Compute the mean used to project onto V_I
    float mu = 0.0;
    int nonZero = 0;
    for (int i = 0; i < n; ++i)
      if ( In_complement[i] > 0 )
      {
        mu += X[i*step];
        ++nonZero;
      }
    mu -= targetMean;
    mu /= nonZero;
        
    // Compute projections onto V_I and X_I together
    // Note that we can test at the same time the convergence:
    // we have converged when all the components of X tilda are positive.
    for (int i = 0; i < n; ++i)
    {
      // Projection onto V_I : enforce the summation to 1
      if ( In_complement[i] > 0 )
        Xtilda[i] = X[i*step] - mu;
      else
        Xtilda[i] = 0.0;
            
      // Projection onto X_I : enforce the positivity
      X[i*step] = MAX(Xtilda[i], 0.0);
            
      // Convergence check
      if ( Xtilda[i] < 0.0 )
      {
        In_complement[i] = 0;
        success = false;
      }
    }
        
    ++iterationCount;
  }
    
    
  // Release memory
  delete []In_complement;
  delete []Xtilda;
}

void lts2::SoftThresholding(cv::Mat& X, float threshold)
{
#ifdef WITH_DISPATCH
  dispatch_queue_t queue = dispatch_get_global_queue(DISPATCH_QUEUE_PRIORITY_HIGH, 0);
  __block cv::Mat X_ = X;
    
  dispatch_apply(X.rows, queue, 
                 ^(size_t y) {
                   float* p_x = X_.ptr<float>(y);
                       
                   for (int x = 0; x < X.cols; ++x, ++p_x)
                   {
                     float absX = std::fabs(*p_x);
                           
                     if ( absX < 1e-6 )
                       *p_x = 0.0;
                     else 
                     {
                       *p_x /= absX;
                       absX -= threshold;
                               
                       if ( absX > 0.0 )
                         *p_x *= absX;
                       else
                         *p_x = 0.0;
                     }
                   }

                 });
#else
  for (int y = 0; y < X.rows; ++y)
  {
    float* p_x = X.ptr<float>(y);
        
    for (int x = 0; x < X.cols; ++x, ++p_x)
    {
      float absX = std::fabs(*p_x);
            
      if ( absX < 1e-6 )
        *p_x = 0.0;
      else 
      {
        *p_x /= absX;
        absX -= threshold;
                
        if (absX > 0.0)
          *p_x *= absX;
        else
          *p_x = 0.0;
      }
    }
  }
#endif
}

void lts2::SoftThresholding(cv::Mat &X1, cv::Mat &X2, float threshold)
{
  for (int y = 0; y < X1.rows; ++y)
  {
    float* p_x1 = X1.ptr<float>(y);
    float* p_x2 = X2.ptr<float>(y);
        
    for (int x = 0; x < X1.cols; ++x, ++p_x1, ++p_x2)
    {
      float absX = hypotf(*p_x1, *p_x2);
            
      if ( absX < 1e-6 )
      {
        *p_x1 = 0.0;
        *p_x2 = 0.0;
      }
      else 
      {
        *p_x1 /= absX;
        *p_x2 /= absX;

        absX -= threshold;
                
        if ( absX > 0.0 )
        {
          *p_x1 *= absX;
          *p_x2 *= absX;
        }
        else
        {
          *p_x1 = 0.0;
          *p_x2 = 0.0;
        }
      }
    }
  }
}
