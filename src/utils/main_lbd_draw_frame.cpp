//  main_lbd_draw_frame.cpp
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
#include <iomanip>
#include <string>
#include <vector>
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
  fprintf(stream, "[-n normalize=0] ");
  fprintf(stream, "[-p psize=32] ");
  fprintf(stream, "N directory\n");
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
  int normalize = 1;

  static struct option long_options[] = {
    {"lbd", required_argument, 0, 0},
    {0, 0, 0, 0}
  };

  int optionIndex;
  int c = getopt_long(argc, argv, "n:p:", long_options, &optionIndex);

  while (c != -1)
  {
    switch (c)
    {
    case 0:
      if (optionIndex == 0)
      {
        if (strncmp(optarg, "freak", 5) == 0) lbdType = (int)lbd::eTypeFreak;
        else if (strncmp(optarg, "brief", 5) == 0) lbdType = (int)lbd::eTypeBrief;
        else if (strncmp(optarg, "exfreak", 7) == 0) lbdType = (int)lbd::eTypeExFreak;
        else lbdType = lbd::eTypeRandomFreak;
      }
      break;
      
    case 'n':
      normalize = atoi(optarg);
      break;

    case 'p':
      pside = atoi(optarg);
      pside += (pside % 2);   // Enforce even size
      break;

    case '?':
      std::cerr << "Unknown option: -" << c << " (ignored)\n";
      break;

    default:
      break;
    }

    c = getopt_long(argc, argv, "n:p:", long_options, &optionIndex);
  }

  int N = atoi(argv[argc-2]);

  cv::Size patchSize(pside,pside);
  lts2::LBDOperator *LBD = lts2::CreateLbdOperator(lbdType);
  LBD->initWithPatchSize(patchSize);

  std::string lbdName;
  switch(lbdType)
  {
  case lbd::eTypeFreak:
    lbdName = "freak";
    break;
  case lbd::eTypeExFreak:
    lbdName = "exfreak";
    break;
  case lbd::eTypeRandomFreak:
    lbdName = "rafreak";
    break;
  case lbd::eTypeBrief:
    lbdName = "brief";
    break;
  }

  std::vector<cv::Mat> frame;
  LBD->drawBasis(frame, normalize);
  cv::Mat res8;

  for (int i = 0; i < N; ++i)
  {
    std::stringstream nameStream;
    nameStream << argv[argc-1] << "/" << lbdName << "-";
    nameStream << std::setw(3) << std::setfill('0') << (i+1);
    nameStream << ".png";

    frame[i].convertTo(res8, CV_8U, 255.0);

    cv::imwrite(nameStream.str(), res8);
  }

  return EXIT_SUCCESS;
}
