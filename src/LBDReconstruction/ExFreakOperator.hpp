//  ExFreakOperator.hpp
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

#ifndef LTS2_EXFREAKOPERATOR_HPP
#define LTS2_EXFREAKOPERATOR_HPP

#include "ILBDOperator.hpp"

#include <opencv2/core/core.hpp>
#include <vector>

namespace lts2
{
  class ExFreakOperator : public LBDOperator
  {
  public:
    ExFreakOperator(int M=512);
    ~ExFreakOperator();

    virtual void initWithPatchSize(cv::Size patchSize);
    
    // Inherited methods
    virtual void Apply(cv::Mat const& X, cv::Mat& Ax);
    virtual float L() const;
    virtual void ReleaseMemory();
  };
}

#endif  // LTS2_EXFREAKOPERATOR_HPP
