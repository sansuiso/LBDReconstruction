//  BriefOperator.cpp
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

#include "BriefOperator.hpp"

#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>

#include <iostream>

#ifdef WITH_DISPATCH
#include <dispatch/dispatch.h>
#endif

lts2::BriefOperator::BriefOperator(int M) : LBDOperator(M) {}

lts2::BriefOperator::~BriefOperator() {}

void lts2::BriefOperator::initWithPatchSize(cv::Size patchSize)
{
    // Set the size variable
    this->setPatchSize(patchSize);

    cv::RNG rng(getpid());

    // Only 9x9 patches (except borders)
    int const detectorRadius = 9 / 2;

    // Reset
    if (_sensitiveCells)
      delete[] _sensitiveCells;

    // Points where we can have a sensitive cell
    int usefulWidth = patchSize.width - 9;
    int usefulHeight = patchSize.height - 9;
    _patternPoints = usefulWidth*usefulHeight;
    _sensitiveCells = new lbd::IntegratingCell[2*_pairsInUse];

    double invCellArea = 1.0/81.0;

    std::cerr << "Generating " << _pairsInUse << " pairs.\n";

    for (int i = 0; i < _pairsInUse; ++i)
    {
      unsigned int x_pos = rng(usefulWidth);
      unsigned int y_pos = rng(usefulHeight);

      _sensitiveCells[2*i].xmin = x_pos;
      _sensitiveCells[2*i].ymin = y_pos;
      _sensitiveCells[2*i].xmax = x_pos + 9;
      _sensitiveCells[2*i].ymax = y_pos + 9;
      _sensitiveCells[2*i].scaleFactor = invCellArea;

      unsigned int x_neg = rng(usefulWidth);
      unsigned int y_neg = rng(usefulHeight);

      _sensitiveCells[2*i+1].xmin = x_neg;
      _sensitiveCells[2*i+1].ymin = y_neg;
      _sensitiveCells[2*i+1].xmax = x_neg + 9;
      _sensitiveCells[2*i+1].ymax = y_neg + 9;
      _sensitiveCells[2*i+1].scaleFactor = invCellArea;

      _basisFunctions[i].pos_cell = 2*i;
      _basisFunctions[i].neg_cell = 2*i+1;
    }
    std::cerr << "Done !\n";
}

void lts2::BriefOperator::Apply(cv::Mat const& X, cv::Mat& Ax)
{
    // Create a column vector where rows = number_of_active_pairs
    Ax.create(_pairsInUse, 1, CV_32FC1);
    Ax.setTo(cv::Scalar(0));

    // Compute the integral image for later
    if (_integralImage.data)
        _integralImage.setTo(cv::Scalar(0));
    cv::integral(X, _integralImage, CV_64F);
    
    // Apply each pair
#ifdef WITH_DISPATCH
    dispatch_queue_t queue = dispatch_get_global_queue(DISPATCH_QUEUE_PRIORITY_HIGH, 0);
    dispatch_apply(_pairsInUse, queue,
                   ^(size_t i) {
                     double mes_pos = measureAtCell(_basisFunctions[i].pos_cell);
                     double mes_neg = measureAtCell(_basisFunctions[i].neg_cell);
                     *Ax.ptr<float>(i) = cv::saturate_cast<float>(mes_pos - mes_neg);
                   });
#else
    float* p_ax = Ax.ptr<float>(0);
    for (int i = 0; i < _pairsInUse; ++i)
    {
        double mes_pos = measureAtCell(_basisFunctions[i].pos_cell);
        double mes_neg = measureAtCell(_basisFunctions[i].neg_cell);
        
        *p_ax++ = cv::saturate_cast<float>(mes_pos - mes_neg);
    }
#endif
}

void lts2::BriefOperator::ApplyConjugate(cv::Mat const& X, cv::Mat& AstarX)
{
    // The output is an image (actually, a patch)
    AstarX.create(_patchSize, CV_32F);
    
    // Stub to hold the current image basis function: we will write
    // each basis function into it
    cv::Mat basisImage(_patchSize, CV_32F, cv::Scalar(0));
    
    // Coordinate value
    float const* p_x = X.ptr<float>(0);
    
    // Sum over the basis functions
    for (int i = 0; i < _pairsInUse; ++i)
    {
        // Reset
        basisImage.setTo(cv::Scalar(0));
        
        // Add "nominal" basis function
        fillBasisFunctionForIndex(basisImage, i);
        
        // Apply the amplitude
        basisImage *= (*p_x);
        ++p_x;
        
        // Sum
        AstarX += basisImage;
    }
}

float lts2::BriefOperator::L() const
{
    return 10.0;
}

void lts2::BriefOperator::ReleaseMemory() {}

lts2::LBDOperator *lts2::CreateBriefOperator(int M)
{
  lts2::BriefOperator *out = new lts2::BriefOperator(M);
  if (!out)
  {
    std::cerr << __FUNCTION__ << "\t" << "Something when wrong :-(";
  }
  return out;
}
