//  main_draw_lbd_circles.cpp
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
#include <opencv2/opencv.hpp>

#include "ILBDOperator.hpp"

int main(int argc, char * const *argv)
{
  int M = 256;
  cv::Size imsize(512, 512);
  cv::Size psize(32,32);

  int lbdType;
  if (strncmp(argv[1], "freak", 5) == 0)
  {
    lbdType = (int)lbd::LBD_TYPE::eTypeFreak;;
  }
  else
  {
    if (strncmp(argv[1], "brief", 5) == 0)
    {
      lbdType = (int)lbd::LBD_TYPE::eTypeBrief;
    }
    else
    {
      std::cerr << "Error. The parameter should be either freak or brief.\n";
      return -1;
    }
  }

  lts2::LBDOperator *LBD = lts2::CreateLbdOperator(lbdType, M);
  LBD->initWithPatchSize(psize);

  cv::Mat destImage(cv::Size(imsize), CV_8UC3);
  LBD->drawSelfAsCircles(destImage);
  
  cv::imwrite(argv[2], destImage);

  return EXIT_SUCCESS;
}
