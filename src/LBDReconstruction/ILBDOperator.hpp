//  ILBDOperator.hpp
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

#ifndef LTS2_ILBDOPERATOR_HPP
#define LTS2_ILBDOPERATOR_HPP

#include "ILinearOperator.hpp"

#include <opencv2/core/core.hpp>
#include <vector>

namespace lbd
{
  unsigned int const kFreakTotalPairs = 641;

  struct BasisFunction {
    unsigned int pos_cell;
    unsigned int neg_cell;
  };
  
  struct IntegratingCell {
    int xmin;
    int xmax;
    int ymin;
    int ymax;
    double scaleFactor;
  };

  enum class LBD_TYPE {
    eTypeFreak       = 0,
    eTypeRandomFreak = 1,
    eTypeBrief       = 2,
    eTypeExFreak     = 3,
  };
}

namespace lts2
{
  class LBDOperator : public LinearOperator
  {
  protected:
    unsigned int _pairsInUse;
  
    cv::Size _patchSize;
    cv::Mat _integralImage;
  
    unsigned int _patternPoints;
    
    lbd::BasisFunction *_basisFunctions;
    lbd::IntegratingCell *_sensitiveCells;
            
    void updateRingCells(lbd::IntegratingCell* cells, int numberOfCells, float detectorSize, float ringRadius, float omega, float phi);
  
    float _L;

  public:
    LBDOperator(int M=512);
    ~LBDOperator();
  
    // Subclasses should implement the following method
    virtual void initWithPatchSize(cv::Size patchSize) = 0;
    void setPatchSize(cv::Size imagePatchSize);
  
    void fillBasisFunctionForIndex(cv::Mat& basisFunction, int idx) const;
    double measureAtCell(int cellIndex) const;
  
    unsigned int pairs() const;  
    void recordBasisMovie(std::string const& movieName);

    void setLipschitzConstant(float L);
    int patchArea() const;

    void drawSelfInImage(cv::Mat &destImage, int normalize=0) const;
    void drawBasis(std::vector<cv::Mat> &basisFunctions, int normalize=0) const;
    void drawSelfAsCircles(cv::Mat &destImage) const;

    // Inherited methods
    virtual void Apply(cv::Mat const& X, cv::Mat& Ax);
    virtual void ApplyConjugate(cv::Mat const& X, cv::Mat& AStarX);
    virtual void ReleaseMemory();
  };

  // Factory methods
  LBDOperator *CreateLbdOperator(int lbdType, int M=512);
  LBDOperator *CreateFreakOperator(int M=512);
  LBDOperator *CreateBriefOperator(int M=512);
  LBDOperator *CreateRandomFreakOperator(int M=512);
  LBDOperator *CreateExFreakOperator(int M=512);
  LBDOperator *CreateLbdOperatorFromYml(std::string const &filename);

}

#endif  // LTS2_ILBDOPERATOR_HPP
