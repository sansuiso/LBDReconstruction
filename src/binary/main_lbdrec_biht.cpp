//  main_lbdrec_biht.cpp
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
#include <getopt.h>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include "ILBDOperator.hpp"
#include "reconstruction.h"

void print_usage(FILE *stream, char **argv)
{
  fprintf(stream, "Usage: %s ", argv[0]);
  fprintf(stream, "[--lbd=freak|brief|rfreak|exfreak] ");
  fprintf(stream, "[--fast] ");
  fprintf(stream, "[--interactive] ");
  fprintf(stream, "[-i iterations=1000] ");
  fprintf(stream, "[-m measures=512] ");
  fprintf(stream, "[-p psize=32] ");
  fprintf(stream, "[-o offset] ");
  fprintf(stream, "[-s sparsity=0.4] ");
  fprintf(stream, "input_image result_image\n");
}

int main(int argc, char **argv)
{
  if (argc < 2)
  {
    print_usage(stdout, argv);
    return 0;
  }

  int lbdType = 0;
  int pside = 32;
  int iterations = 1000;
  float sparsity = 0.4;
  int offset = 0;
  int process_only_fast = 0;
  int interactive = 0;
  int measures = 512;

  static struct option long_options[] = {
    {"lbd", required_argument, 0, 0},
    {"fast", no_argument, &process_only_fast, 1},
    {"interactive", no_argument, &interactive, 1},
    {0, 0, 0, 0}
  };

  int optionIndex;
  int c = getopt_long(argc, argv, "i:m:o:p:s:", long_options, &optionIndex);
  
  while (c != -1)
  {
    switch (c)
    {
    case 0:
      if (optionIndex == 0)
      {
        if (strncmp(optarg, "freak", 5) == 0) lbdType = (int)lbd::LBD_TYPE::eTypeFreak;
        else if (strncmp(optarg, "brief", 5) == 0) lbdType = (int)lbd::LBD_TYPE::eTypeBrief;
        else if (strncmp(optarg, "exfreak", 7) == 0) lbdType = (int)lbd::LBD_TYPE::eTypeExFreak;
        else lbdType = (int)lbd::LBD_TYPE::eTypeRandomFreak;
      }
      break;
      
    case 'i':
      iterations = atoi(optarg);
      break;

    case 'm':
      measures = atoi(optarg);
      break;

    case 'p':
      pside = atoi(optarg);
      pside += (pside % 2);   // Enforce even size
      break;

    case 'o':
      offset = atoi(optarg);
      break;

    case 's':
      sparsity = (float)atof(optarg);
      break;

    case '?':
      std::cerr << "Unknown option: -" << c << " (ignored)\n";
      break;

    default:
      break;
    }

    c = getopt_long(argc, argv, "i:m:o:p:s:", long_options, &optionIndex);
  }

  cv::Mat testImage = cv::imread(argv[argc-2], 0);
  if (!testImage.data)
  {
    std::cerr << "Error opening image: " << argv[argc-2] << std::endl;
    print_usage(stderr, argv);
    return EXIT_FAILURE;
  }

  cv::Mat testImagef;
  testImage.convertTo(testImagef, CV_32F, 1.0/255.0);

  cv::Size patchSize(pside,pside);
  if (offset == 0) offset = pside;
  cv::Point patchOffset(offset,offset);

  lts2::LBDOperator *LBD = lts2::CreateLbdOperator(lbdType, measures);
  LBD->initWithPatchSize(patchSize);

  cv::Mat result;

  if (process_only_fast == 0)
    lts2::PerformBIHTOnImage(testImagef, patchSize, patchOffset,
                            *LBD, result, iterations, sparsity, -1.0);
  else
    lts2::PerformBIHTOnImageFAST(testImagef, patchSize, *LBD, result,
                                iterations, sparsity, -1.0);

  std::stringstream windowNameStr;
  windowNameStr << "BIHT" << " ";
  if (process_only_fast != 0)
    windowNameStr << "FAST " << " ";

  switch (lbdType)
  {
  case (int)lbd::LBD_TYPE::eTypeFreak:
    windowNameStr << "FREAK";
    break;

  case (int)lbd::LBD_TYPE::eTypeRandomFreak:
    windowNameStr << "Randomized FREAK";
    break;

  case (int)lbd::LBD_TYPE::eTypeBrief:
    windowNameStr << "BRIEF";
    break;

  case (int)lbd::LBD_TYPE::eTypeExFreak:
    windowNameStr << "Ex-FREAK";
    break;
  }

  cv::Mat res8;
  result.convertTo(res8, CV_8U, 255.0);

  if (interactive > 0)
  {
    cv::imshow(windowNameStr.str(), res8);
    cv::waitKey();
  }

  cv::imwrite(argv[argc-1], res8);

  return EXIT_SUCCESS;
}
