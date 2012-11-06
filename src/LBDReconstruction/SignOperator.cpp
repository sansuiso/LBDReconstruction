//  SignOperator.cpp
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

#include "SignOperator.hpp"

#ifdef WITH_DISPATCH
#include <dispatch/dispatch.h>
#endif

#define MY_EPSILON 1e-6

lts2::SignOperator::SignOperator() {}

lts2::SignOperator::~SignOperator() {}

void lts2::SignOperator::Apply(cv::Mat const &X, cv::Mat &Ax)
{
  CV_Assert(X.type() == CV_32F && X.data);

  Ax.create(X.size(), X.type());

#ifdef WITH_DISPATCH
  dispatch_queue_t queue = dispatch_get_global_queue(DISPATCH_QUEUE_PRIORITY_HIGH, 0);
  dispatch_apply(Ax.rows, queue,
                 ^(size_t y) {
                   float const *p_x = X.ptr<float>(y);
                   float *p_ax = Ax.ptr<float>(y);

                   for (int x = 0; x < Ax.cols; ++x, ++p_x, ++p_ax)
                   {
                     if (*p_x > MY_EPSILON) *p_ax = 1.0;
                     else if (*p_x < -MY_EPSILON) *p_ax = -1.0;
                     else *p_ax = 0.0;
                   }
                 });
#else
  for (int y = 0; y < Ax.rows; ++y)
  {
    float const *p_x = X.ptr<float>(y);
    float *p_ax = Ax.ptr<float>(y);

    for (int x = 0; x < Ax.cols; ++x, ++p_x, ++p_ax)
    {
      if (*p_x > MY_EPSILON) *p_ax = 1.0;
      else if (*p_x < -MY_EPSILON) *p_ax = -1.0;
      else *p_ax = 0.0;
    }
  }
#endif
}

void lts2::SignOperator::ApplyConjugate(cv::Mat const &X, cv::Mat &AstarX)
{
  return this->Apply(X, AstarX);
}

float lts2::SignOperator::L() const
{
  return 0.0;
}

void lts2::SignOperator::ReleaseMemory() {}
