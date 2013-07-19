//  main_lbd_merge2movie.cpp
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
#include <opencv2/opencv.hpp>

int main(int argc, char const * const argv[])
{
  if (argc < 4)
  {
    std::cerr << "Missing arguments! 2 images + 1 movie name are required.\n";
    return EXIT_FAILURE;
  }

  // Load 2 images
  cv::Mat I0 = cv::imread(argv[argc-3], 0);
  if (!I0.data)
  {
    std::cerr << "Error reading: " << argv[argc-3] << std::endl;
    return EXIT_FAILURE;
  }

  cv::Mat I1 = cv::imread(argv[argc-2], 0);
  if (!I1.data)
  {
    std::cerr << "Error reading: " << argv[argc-2] << std::endl;
    return EXIT_FAILURE;
  }

  CV_Assert(I0.size() == I1.size());

  // Output movie
  double fps = 1.25;
  int fourcc = cv::VideoWriter::fourcc('m','p','4','v');
  cv::Size frameSize = I0.size();

  cv::VideoWriter movie(argv[argc-1], fourcc, fps, frameSize, true);
  CV_Assert(movie.isOpened());
  
  float mixCoeff = 0.0;
  float mixStep = 0.1;
  cv::Mat mixf, mixc, mix8;

  cv::Mat I0f;
  I0.convertTo(I0f, CV_32F, 1.0/255.0);
  cv::Mat I1f;
  I1.convertTo(I1f, CV_32F, 1.0/255.0);

  std::cout << "Forward...\n";
  while (mixCoeff <= 1.0)
  {
    mixf = (1.0-mixCoeff)*I0f + mixCoeff*I1f;
    mixf.convertTo(mix8, CV_8U, 255.0);
    cv::cvtColor(mix8, mixc, cv::COLOR_GRAY2BGR);

    movie << mixc;

    mixCoeff += mixStep;
  }

  mixCoeff = 1.0;
  std::cout << "Backward...\n";
  while (mixCoeff >= 0.0)
  {
    mixf = (1.0-mixCoeff)*I0f + mixCoeff*I1f;
    mixf.convertTo(mix8, CV_8U, 255.0);
    cv::cvtColor(mix8, mixc, cv::COLOR_GRAY2BGR);

    movie << mixc;

    mixCoeff -= mixStep;
  }
}
