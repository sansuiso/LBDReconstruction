//  FreakOperator.cpp
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

#include "FreakOperator.hpp"
#include "freak.h"

#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>

#include <iostream>

#ifdef WITH_DISPATCH
#include <dispatch/dispatch.h>
#endif

/**
 * How many measures along the ring ? 
 */
extern unsigned int const FREAKS_CELLS_PER_RING[];

/**
 * Rankings from the training phase. Each index refers to a test, ie a (+) and a (-) measurement.
 */
extern unsigned int const FREAKS_RANKED_PAIRS[];

lts2::FreakOperator::FreakOperator(int M) : LBDOperator(M) {}

lts2::FreakOperator::~FreakOperator() {}

void lts2::FreakOperator::initWithPatchSize(cv::Size patchSize)
{
    // Set the size variable
    this->setPatchSize(patchSize);
    
    int patchRadius = MIN(_patchSize.width, _patchSize.height);
    
    // Update the filter sizes
    // unit = (smallest_radius - biggest_radius) / 21
    float unitSpace = 22.0f * patchRadius / 1575.f;
    
    float ringRadius[8];
    ringRadius[0] = patchRadius / 25;
    ringRadius[1] = ringRadius[0] + 1  * unitSpace; //
    ringRadius[2] = ringRadius[0] + 3  * unitSpace; //
    ringRadius[3] = ringRadius[0] + 6  * unitSpace; //
    ringRadius[4] = ringRadius[0] + 10 * unitSpace; // 
    ringRadius[5] = ringRadius[0] + 15 * unitSpace; // 
    ringRadius[6] = ringRadius[0] + 21 * unitSpace; //
    ringRadius[7] = 0;                              // Center point
    
    float filterSigma[8];
    filterSigma[0] = ringRadius[0] / 2;
    filterSigma[1] = ringRadius[1] / 2;
    filterSigma[2] = ringRadius[2] / 2;
    filterSigma[3] = ringRadius[3] / 2;
    filterSigma[4] = ringRadius[4] / 2;
    filterSigma[5] = ringRadius[5] / 2;
    filterSigma[6] = ringRadius[6] / 2;
    filterSigma[7] = ringRadius[0] / 3; // Center point

#ifdef DEBUG
    for (int i = 0; i < 8; ++i)
        std::cerr << "ring[" << i << "] = " << ringRadius[i] << " (" << filterSigma[i] << ")" << std::endl;
#endif

    // Create the measurement points
    _patternPoints = 0;
    for (int ring = 0; ring < 8; ++ring)
        _patternPoints += FREAKS_CELLS_PER_RING[ring];
    
    // Reset
    if (_sensitiveCells)
        delete[] _sensitiveCells;
    _sensitiveCells = new lbd::IntegratingCell[_patternPoints];
    
    lbd::IntegratingCell* p_cells = _sensitiveCells;
    for (unsigned int ring = 0; ring < 8; ++ring)
    {
        float phi = M_PI / FREAKS_CELLS_PER_RING[ring] * ring;
        float omega = 2.0 * M_PI / FREAKS_CELLS_PER_RING[ring];
        
        updateRingCells(p_cells, FREAKS_CELLS_PER_RING[ring], filterSigma[ring], 
                        ringRadius[ring], omega, phi);
        
        p_cells += FREAKS_CELLS_PER_RING[ring];
    }
    
    // Create the possible pairs
    lbd::BasisFunction* allPairs = new lbd::BasisFunction[_patternPoints*(_patternPoints-1)/2];
    for (unsigned int j = 1, k = 0; j < _patternPoints; ++j)
        for (unsigned i = 0; i < j; ++i, ++k)
        {
            allPairs[k].pos_cell = i;
            allPairs[k].neg_cell = j;
        }
    
    // Now, retain only the best ones
    for (unsigned int i = 0; i < _pairsInUse; ++i)
      _basisFunctions[i] = allPairs[ FREAKS_RANKED_PAIRS[i % lbd::kFreakTotalPairs] ];
    
    // Not needed anymore
    delete[] allPairs;
}

void lts2::FreakOperator::Apply(cv::Mat const& X, cv::Mat& Ax)
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

void lts2::FreakOperator::ApplyConjugate(cv::Mat const& X, cv::Mat& AstarX)
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

float lts2::FreakOperator::L() const
{
//    return _L;
    return 10.0;
}

void lts2::FreakOperator::ReleaseMemory() {}

lts2::LBDOperator *lts2::CreateFreakOperator(int M)
{
  lts2::FreakOperator *out = new lts2::FreakOperator(M);
  if (!out)
  {
    std::cerr << __FUNCTION__ << "\t" << "Something when wrong :-(";
  }
  return out;
}
