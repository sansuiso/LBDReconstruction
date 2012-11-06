//  ILinearOperator.hpp
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

#ifndef LTS2_ILINEAROPERATOR_HPP
#define LTS2_ILINEAROPERATOR_HPP

#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>

namespace lts2 
{
  class LinearOperator
  {
  public:
    LinearOperator() : _norm(1.0) {}
    virtual ~LinearOperator() {}

    void setNorm(float norm) { 
      _norm = norm; 
    }
    
    float norm() const { 
      return _norm; 
    }
    
    virtual void Apply(cv::Mat const& X, cv::Mat& Ax) = 0;
    virtual void ApplyConjugate(cv::Mat const& X, cv::Mat& AStarX) = 0;

    virtual void RelaseMemory() {}
    
    /**
     * Computes an approximate value for the norm of an operator 
     * based on the power method.
     * @brief Estimate the Lipschitz norm of a linear operator
     * @param inputSize Size of the input image data (will be of type CV_32FC1, bounded by [0, 1]
     * @param iterations Desired number of iterations (the more the better)
     * @returns Estimate of the norm
     */
    static 
    float EstimateOperatorNorm(LinearOperator& ope, cv::Size const& inputSize, int iterations) {
      cv::Mat X, Ax;
      float norm = 1.0f;
      
      cv::RNG rng(getpid());
      rng.fill(X, cv::RNG::UNIFORM, 0.0f, 1.0f);
      X /= cv::norm(X);

      return LinearOperator::EstimateOperatorNorm(ope, X, iterations);
    }

    static
    float EstimateOperatorNorm(LinearOperator &ope, cv::Mat &inputPoint, int iterations) {
      cv::Mat X, Ax;
      float norm = 1.0f;

      X = inputPoint.clone();

      for (int i = 0; i < iterations; ++i)
      {
        // K^\star * K * X   
        ope.Apply(X, Ax);
        ope.ApplyConjugate(Ax, X);

        // Normalize X
        float norm_x = MAX(cv::norm(X), 1e-6);
        X /= norm_x;

        // K * X
        ope.Apply(X, Ax);

        // Update estimate
        norm = cv::norm(Ax);
      }

      return norm;
    }

  protected:
    float _norm;
  };
  
}

#endif  // LTS2_ILINEAROPERATOR_HPP
