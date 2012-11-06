//  main_lbd_merge2image.cpp
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

#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>
#include <getopt.h>

int main(int argc, char * const argv[])
{
  if (argc < 4)
  {
    std::cerr << "Missing arguments!\n";
    std::cerr << "Usage: " << argv[0] << " lbd_image image_to_merge result" << std::endl;
    return EXIT_FAILURE;
  }

  float threshold = 0.2;
  int sobelSize = 3;
  int dash = 0;
  float alpha = 0.7;

  int c = 0;
  while ( (c = getopt(argc, argv, "a:ds:t:")) != -1 )
  {
    switch (c)
    {
    case 'a':
      alpha = (float)atof(optarg);
      break;

    case 'd':
      dash = 1;
      break;

    case 's':
      sobelSize = atoi(optarg);
      break;

    case 't':
      threshold = (float)atof(optarg);
      threshold = MIN(MAX(0.0, threshold), 1.0);
      break;

    default:
      break;
    }
  }

  // Read LBD image
  cv::Mat lbd = cv::imread(argv[argc-3], 0);
  if (!lbd.data)
  {
    std::cerr << "Error reading; " << argv[argc-3] << std::endl;
    return EXIT_FAILURE;
  }

  // Read src image
  cv::Mat src = cv::imread(argv[argc-2], 0);
  if (!src.data)
  {
    std::cerr << "Error reading; " << argv[argc-2] << std::endl;
    return EXIT_FAILURE;
  }

  if (lbd.size() != src.size())
  {
    cv::Mat tmp;
    cv::resize(src, tmp, lbd.size());
    tmp.copyTo(src);
  }

  // Take gradient
  cv::Mat dx, dy;
  cv::Sobel(src, dx, CV_32F, 1, 0, sobelSize);
  cv::Sobel(src, dy, CV_32F, 0, 1, sobelSize);
  cv::multiply(dx, dx, dx);
  cv::multiply(dy, dy, dy);
  cv::Mat edges;
  edges = dx + dy;
  cv::sqrt(edges, edges);
  edges = cv::abs(edges);
  cv::normalize(edges, edges, 0, 1, cv::NORM_MINMAX);

  cv::Mat sobel;
  edges.convertTo(sobel, CV_8U, 255.0);

  // Create mask if needed
  cv::Mat dashMask(lbd.size(), CV_32FC1, cv::Scalar(1.0));
  if (dash > 0)
  {
    for (int y = 0; y < dashMask.rows; ++y)
    {
      int testValue = (y/2) % 2;
      float *p_dash = dashMask.ptr<float>(y);

      for (int x = 0; x < dashMask.cols; ++x)
        *p_dash++ *= ((x/2)%2 == testValue);
    }
  }
  cv::multiply(edges, dashMask, edges);

  // Merge step
  std::vector<cv::Mat> planes(3);
  for (int i = 0; i < 3; ++i)
    lbd.convertTo(planes[i], CV_32F, 1.0/255.0);

  int width = lbd.cols;
  int height = lbd.rows;

  for (int i = 0; i < 3; ++i)
    planes[i] = 1.0*planes[i] + 0.0*edges;

  for (int y = 0; y < height; ++y)
  {
    float const *p_edges = edges.ptr<float>(y);
    float *p_blue = planes[0].ptr<float>(y);
    float *p_green = planes[1].ptr<float>(y);
    float *p_red = planes[2].ptr<float>(y);

    for (int x = 0; x < width; ++x)
    {
      if (*p_edges >= threshold)
      {
        *p_red = alpha + (1.0-alpha)*(*p_red);
        *p_green = alpha + (1.0-alpha)*(*p_green);
      }
      p_blue++;
      p_green++;
      p_red++;
      p_edges++;
    }
  }

  cv::Mat colorf, color;
  cv::merge(planes, colorf);
  colorf.convertTo(color, CV_8U, 255.0);
  
  // Done
  cv::imwrite(argv[argc-1], color);

  return EXIT_SUCCESS;
}
