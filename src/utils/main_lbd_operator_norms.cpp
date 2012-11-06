//  main_lbd_operator_norms.cpp
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
#include <string>
#include <vector>
#include <getopt.h>

#include <opencv2/opencv.hpp>

#include "ILinearOperator.hpp"
#include "ILBDOperator.hpp"

int main(int argc, char *const *argv)
{
  std::vector<std::string> lbdNames;
  lbdNames.push_back("Freak");
  lbdNames.push_back("Random Freak");
  lbdNames.push_back("Brief");
  lbdNames.push_back("Ex. Freak");

  int measures = 512;
  int psize = 32;
  int iterations = 100;
  int samples = 20;

  int c = 0;
  while( (c = getopt(argc, argv, "i:p:m:s:")) != -1)
  {
    switch(c)
    {
    case 'i':
      iterations = atoi(optarg);
      break;
    case 'p':
      psize = atoi(optarg);
      break;
    case 'm':
      measures = atoi(optarg);
      break;
    case 's':
      samples = atoi(optarg);
      break;
    default:
      break;
    }
  }

  cv::Mat testImage = cv::imread(argv[argc-1], 0);
  if (!testImage.data)
  {
    std::cerr << "Error reading: " << argv[argc-1] << std::endl;
    return -1;
  }

  cv::Mat imagef;
  testImage.convertTo(imagef, CV_32F, 1.0/255.0);

  cv::RNG rng(getpid());

  cv::Size patchSize(psize,psize);

  for (int i = 0; i < lbdNames.size(); ++i)
  {
    lts2::LBDOperator *LBD = lts2::CreateLbdOperator(i, measures);
    LBD->initWithPatchSize(patchSize);
    float norm = 0.0;

    for (int k = 0; k < samples; ++k)
    {
      // Select a random patch
      cv::Rect ROI;
      ROI.x = rng(imagef.cols-psize);
      ROI.y = rng(imagef.rows-psize);
      ROI.width = patchSize.width;
      ROI.height = patchSize.height;

      cv::Mat patch;
      imagef(ROI).copyTo(patch);

      // Compute the norm
      float current_norm = lts2::LinearOperator::EstimateOperatorNorm(*LBD, patch, iterations);

      norm = MAX(norm, current_norm);
    }

    std::cout << lbdNames[i] << " has norm: " << norm << std::endl;
    delete LBD;
  }

  return EXIT_SUCCESS;
}
