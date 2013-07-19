//  ILBDOperator.cpp
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

#include "ILBDOperator.hpp"

#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>

#include <iostream>

lts2::LBDOperator::LBDOperator(int M) : lts2::LinearOperator(), _pairsInUse(M), _patchSize(cv::Size(64,64)), _sensitiveCells(NULL), _basisFunctions(NULL)
{
  this->setNorm(10.0);
  _basisFunctions = new lbd::BasisFunction[_pairsInUse];
}

lts2::LBDOperator::~LBDOperator()
{
    if (_sensitiveCells)
        delete[] _sensitiveCells;
    if (_basisFunctions)
      delete[]_basisFunctions;
}

void lts2::LBDOperator::setPatchSize(cv::Size imagePatchSize)
{
    _patchSize = imagePatchSize;
}

void lts2::LBDOperator::updateRingCells(lbd::IntegratingCell* cells, int numberOfCells, float detectorSize, float ringRadius, float omega, float phi)
{
    float xoffset = _patchSize.width / 2;
    float yoffset = _patchSize.height / 2;
    
    for (int cellIdx = 0; cellIdx < numberOfCells; ++cellIdx)
    {
      lbd::IntegratingCell* currentCell = cells + cellIdx;
        
        float theta = cellIdx * omega + phi;
        
        float xc = xoffset + ringRadius * std::cos(theta);
        float yc = yoffset + ringRadius * std::sin(theta);
        
        int xmin =  (int)lrintf(xc-detectorSize);
        int xmax =  (int)lrintf(xc+detectorSize) + 1;
        int ymin =  (int)lrintf(yc-detectorSize);
        int ymax =  (int)lrintf(yc+detectorSize) + 1;
        
        xmin = MAX(xmin, 0);
        xmax = MIN(xmax, _patchSize.width);
        ymin = MAX(ymin, 0);
        ymax = MIN(ymax, _patchSize.height);
        
        currentCell->xmin = xmin;
        currentCell->xmax = xmax;
        currentCell->ymin = ymin;
        currentCell->ymax = ymax;
        
        double cellArea = double((xmax - xmin)*(ymax - ymin));
        currentCell->scaleFactor = 1.0 / cellArea;
    }
}

void lts2::LBDOperator::fillBasisFunctionForIndex(cv::Mat &basisFunction, int idx) const
{
    // Output
    basisFunction.create(_patchSize, CV_32FC1);
    basisFunction.setTo(cv::Scalar(0));
    
    // Local structures needed
    lbd::IntegratingCell cell;
    int cellId;
    cv::Mat ROI;
    float value;

    //---------------//
    // Positive part //
    //---------------//
    
    cv::Mat posImg(_patchSize, CV_32F, cv::Scalar(0));
    cellId = _basisFunctions[idx].pos_cell;
    cell = _sensitiveCells[cellId];
    value = cv::saturate_cast<float>(cell.scaleFactor);
    
    // Extract ROI
    ROI = posImg(cv::Rect(cell.xmin, cell.ymin, 
                          cell.xmax-cell.xmin, cell.ymax-cell.ymin));
    
    // Set ROI to the correct value
    ROI.setTo(cv::Scalar(value));

    //---------------//
    // Negative part //
    //---------------//

    cv::Mat negImg(_patchSize, CV_32F, cv::Scalar(0));
    cellId = _basisFunctions[idx].neg_cell;
    cell = _sensitiveCells[cellId];
    value = cv::saturate_cast<float>(cell.scaleFactor);

    // Extract ROI
    ROI = negImg(cv::Rect(cell.xmin, cell.ymin, 
                          cell.xmax-cell.xmin, cell.ymax-cell.ymin));
    
    // Set ROI to the correct value
    ROI.setTo(cv::Scalar(value));
    
    //--------------//
    // Combine both //
    //--------------//
    basisFunction = posImg - negImg;
}

double lts2::LBDOperator::measureAtCell(int cellIndex) const
{
    double measure = 0.0;
    
    lbd::IntegratingCell cell = _sensitiveCells[cellIndex];

    double A, B, C, D;
    
    A = *(_integralImage.ptr<double>(cell.ymin) + cell.xmin);
    B = *(_integralImage.ptr<double>(cell.ymin) + cell.xmax);
    C = *(_integralImage.ptr<double>(cell.ymax) + cell.xmin);
    D = *(_integralImage.ptr<double>(cell.ymax) + cell.xmax);
    
    measure = cell.scaleFactor * (D + A - B - C);
    
    return measure;
}

void lts2::LBDOperator::drawSelfInImage(cv::Mat &destImage, int normalize) const
{
  destImage.create(_patchSize, CV_32FC1);
  destImage.setTo(cv::Scalar(0));

  cv::Mat currentBasisVector;
  for (int i = 0; i < _pairsInUse; ++i)
  {
    this->fillBasisFunctionForIndex(currentBasisVector, i);
    currentBasisVector = cv::abs(currentBasisVector);

    if (normalize > 0) cv::threshold(currentBasisVector, currentBasisVector, 0.0, 1.0, cv::THRESH_BINARY);
    if (normalize < 0) cv::normalize(currentBasisVector, currentBasisVector, 0.0, 1.0, cv::NORM_MINMAX);

    destImage += currentBasisVector;
  }
}

void lts2::LBDOperator::drawBasis(std::vector<cv::Mat> &basisFunctions, int normalize) const
{
  basisFunctions.resize(_pairsInUse);
  cv::Mat negPart;

  for (int i = 0; i < _pairsInUse; ++i)
  {
    this->fillBasisFunctionForIndex(basisFunctions[i], i);

    if (normalize > 0) {
      cv::threshold(basisFunctions[i], negPart, -FLT_MIN, -1.0, cv::THRESH_BINARY_INV);
      cv::threshold(basisFunctions[i], basisFunctions[i], FLT_MIN, 1.0, cv::THRESH_BINARY);
      basisFunctions[i] += negPart;
    }
    if (normalize != 0) cv::normalize(basisFunctions[i], basisFunctions[i], 0.0, 1.0, cv::NORM_MINMAX);
  }
}

void lts2::LBDOperator::drawSelfAsCircles(cv::Mat &destImage) const
{
  if (destImage.size() == cv::Size(0,0))
    destImage.create(_patchSize, CV_8UC3);
  
  // Fill with white
  destImage.setTo(cv::Scalar::all(255));

  // Shortcuts
  cv::Scalar red = cv::Scalar(0, 0, 255);
  cv::Scalar black = cv::Scalar(0, 0, 0);

  // Loop over the cells
  for (int i = 0; i < this->pairs(); ++i)
  {
    cv::Point center;
    int radius, xmin, xmax, ymin, ymax;
    float n_xmin, n_xmax, n_ymin, n_ymax;

    lbd::IntegratingCell plus = _sensitiveCells[_basisFunctions[i].pos_cell];
    lbd::IntegratingCell minus = _sensitiveCells[_basisFunctions[i].neg_cell];

    // (+) first
    n_xmin = plus.xmin / (float)_patchSize.width;
    n_xmax = (plus.xmax-1) / (float)_patchSize.width;
    n_ymin = plus.ymin / (float)_patchSize.height;
    n_ymax = (plus.ymax -1)/ (float)_patchSize.height;

    xmin = rint(n_xmin * destImage.cols);
    xmax = rint(n_xmax * destImage.cols);
    ymin = rint(n_ymin * destImage.rows);
    ymax = rint(n_ymax * destImage.rows);

    radius = MIN(ymax-ymin,xmax-xmin) / 2;
    center.x = xmin + radius;
    center.y = ymin + radius;

    cv::circle(destImage, center, radius, red);
    cv::circle(destImage, center, 1, black, -1);

    // (-) now
    n_xmin = minus.xmin / (float)_patchSize.width;
    n_xmax = (minus.xmax-1) / (float)_patchSize.width;
    n_ymin = minus.ymin / (float)_patchSize.height;
    n_ymax = (minus.ymax -1)/ (float)_patchSize.height;

    xmin = rint(n_xmin * destImage.cols);
    xmax = rint(n_xmax * destImage.cols);
    ymin = rint(n_ymin * destImage.rows);
    ymax = rint(n_ymax * destImage.rows);

    radius = MIN(ymax-ymin,xmax-xmin) / 2;
    center.x = xmin + radius;
    center.y = ymin + radius;

    cv::circle(destImage, center, radius, red);
    cv::circle(destImage, center, 1, black, -1);
  }
}

unsigned int lts2::LBDOperator::pairs() const
{
    return _pairsInUse;
}

void lts2::LBDOperator::Apply(cv::Mat const& X, cv::Mat& Ax)
{
    // Create a column vector where rows = number_of_active_pairs
    Ax.create(_pairsInUse, 1, CV_32FC1);
    Ax.setTo(cv::Scalar(0));

    // Compute the integral image for later
    if (_integralImage.data)
        _integralImage.setTo(cv::Scalar(0));
    cv::integral(X, _integralImage, CV_64F);
    
    // Apply each pair
    float* p_ax = Ax.ptr<float>(0);
    for (int i = 0; i < _pairsInUse; ++i)
    {
        double mes_pos = measureAtCell(_basisFunctions[i].pos_cell);
        double mes_neg = measureAtCell(_basisFunctions[i].neg_cell);
        
        *p_ax++ = cv::saturate_cast<float>(mes_pos - mes_neg);
    }
    
}

void lts2::LBDOperator::ApplyConjugate(cv::Mat const& X, cv::Mat& AstarX)
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

void lts2::LBDOperator::ReleaseMemory() {}

void lts2::LBDOperator::recordBasisMovie(std::string const& movieName)
{
  cv::VideoWriter movie(movieName, cv::VideoWriter::fourcc('m', 'p', '4', 'v'), 2.0, _patchSize);
    
    cv::Mat currrentFrame, grayFrame;
    cv::Mat basisImage(_patchSize, CV_32F, cv::Scalar(0));
    
    double bmin, bmax;
    
    for (int i = 0; i < _pairsInUse; ++i)
    {
        // Fill basis image
        fillBasisFunctionForIndex(basisImage, i);
        cv::minMaxLoc(basisImage, &bmin, &bmax);
        
        // Convert to uchar
        basisImage.convertTo(grayFrame, CV_8U, 255.0/(bmax-bmin), -255.0*bmin/(bmax-bmin));
        
        // Color
        cv::cvtColor(grayFrame, currrentFrame, cv::COLOR_GRAY2BGR);
        
        movie << currrentFrame;
    }
}

void lts2::LBDOperator::setLipschitzConstant(float L)
{
    _L = L;
}

int lts2::LBDOperator::patchArea() const
{
  return _patchSize.area();
}

lts2::LBDOperator *lts2::CreateLbdOperator(int lbdType, int M)
{
  switch (lbdType)
  {
  case lbd::eTypeExFreak:
    return lts2::CreateExFreakOperator(M);
    break;

  case lbd::eTypeBrief:
    return lts2::CreateBriefOperator(M);
    break;

  case lbd::eTypeRandomFreak:
    return lts2::CreateRandomFreakOperator(M);
    break;

  case lbd::eTypeFreak:
  default:
    return lts2::CreateFreakOperator(M);
    break;
  }
}

lts2::LBDOperator *lts2::CreateLbdOperatorFromYml(std::string const &filename)
{
  cv::FileStorage fs(filename, cv::FileStorage::READ);

  if (!fs.isOpened())
  {
    std::cerr << "Invalid file: " << filename << std::endl;
    return NULL;
  }

  LBDOperator *out = NULL;

  int lbdType = 0;
  int M = 512;
  int pwidth = 32;
  int pheight = 32;

  fs["lbd"] >> lbdType;
  fs["measures"] >> M;
  fs["width"] >> pwidth;
  fs["height"] >> pheight;

  out = lts2::CreateLbdOperator(lbdType, M);
  if (out)
  {
    cv::Size patchSize(pwidth, pheight);
    out->initWithPatchSize(patchSize);
  }

  return out;
}
