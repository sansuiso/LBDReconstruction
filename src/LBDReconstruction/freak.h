//  freak.h
//
//	Copyright (C) 2011-2012  Signal Processing Laboratory 2 (LTS2), EPFL,
//	Emmanuel d'Angelo (emmanuel.dangelo@epfl.ch),
//	Laurent Jacques (laurent.jacques@uclouvain.be)
//	Alexandre Alahi (alahi@stanford.edu)
//  Raphael Ortiz (raphael.ortiz@a3.epfl.ch)
//  Kirell Benzi (kirell.benzi@epfl.ch)
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

#ifndef LBDREC_FREAK_H
#define LBDREC_FREAK_H

/**
 * How many measures along the ring ?
 */
unsigned int const FREAKS_CELLS_PER_RING[] = { 6, 6, 6, 6, 6, 6, 6, 1 };

/**
 * Rankings from the training phase. Each index refers to a test, ie a (+) and a (-) measurement.
 */
unsigned int const FREAKS_RANKED_PAIRS[] = {
  654, 503, 302, 100, 517, 869, 495, 627, 236, 857, 594, 275, 818, 232, 254,
  870, 768, 44, 602, 118, 209, 304, 616, 99, 217, 160, 228, 258, 613, 514,
  656, 11, 61, 433, 506, 52, 349, 8, 220, 886, 132, 192, 605, 540, 573, 770,
  142, 379, 401, 163, 480, 448, 544, 82, 478, 173, 383, 588, 194, 147, 775,
  319, 394, 400, 450, 320, 489, 175, 577, 390, 557, 461, 774, 121, 753, 109,
  107, 660, 80, 315, 732, 469, 313, 661, 757, 807, 555, 713, 459, 67, 269,
  363, 439, 291, 819, 809, 125, 846, 645, 643, 437, 590, 467, 491, 529, 71,
  779, 790, 827, 690, 562, 684, 338, 730, 787, 340, 856, 418, 289, 243, 673,
  533, 203, 898, 710, 634, 676, 329, 841, 838, 902, 701, 280, 367, 702, 692,
  141, 186, 566, 801, 278, 247, 742, 737, 180, 632, 422, 738, 813, 830, 844,
  852, 31, 812, 721, 137, 55, 265, 551, 687, 751, 792, 746, 327, 39, 407, 352,
  556, 724, 205, 94, 583, 15, 356, 696, 589, 184, 851, 198, 748, 715, 455,
  361, 484, 405, 136, 719, 20, 199, 798, 93, 764, 309, 413, 416, 879, 179,
  538, 237, 95, 637, 796, 584, 411, 681, 308, 670, 835, 490, 460, 185, 763,
  434, 833, 38, 640, 855, 679, 550, 814, 264, 204, 571, 824, 762, 781, 568,
  374, 648, 248, 769, 324, 32, 485, 822, 698, 283, 651, 735, 56, 270, 384,
  704, 358, 389, 736, 335, 259, 427, 776, 773, 242, 350, 668, 815, 525, 332,
  759, 454, 535, 314, 708, 442, 650, 785, 428, 649, 296, 665, 901, 395, 899,
  621, 343, 622, 662, 582, 475, 472, 369, 174, 850, 520, 344, 655, 286, 858,
  445, 193, 697, 593, 297, 808, 303, 817, 483, 152, 546, 731, 628, 453, 456,
  549, 486, 579, 87, 444, 60, 388, 474, 231, 802, 16, 473, 148, 119, 104,
  726, 626, 740, 113, 443, 500, 294, 526, 629, 253, 567, 126, 378, 860, 560,
  385, 534, 30, 578, 40, 803, 424, 597, 572, 429, 372, 845, 479, 373, 539,
  76, 691, 449, 545, 624, 346, 639, 747, 625, 758, 77, 840, 725, 884, 685,
  695, 596, 638, 295, 608, 599, 607, 214, 527, 523, 752, 508, 511, 5, 10,
  155, 241, 847, 345, 778, 859, 285, 644, 307, 9, 849, 512, 619, 501, 522,
  284, 0, 610, 127, 468, 310, 334, 686, 727, 867, 739, 263, 399, 260, 222,
  839, 611, 509, 114, 25, 86, 498, 438, 693, 290, 806, 497, 130, 112, 333,
  46, 157, 816, 73, 729, 81, 154, 853, 396, 561, 357, 700, 528, 215, 108,
  707, 623, 238, 168, 854, 832, 699, 600, 524, 362, 412, 339, 368, 211, 321,
  741, 633, 804, 47, 423, 88, 417, 318, 26, 675, 178, 733, 131, 223, 120,
  200, 169, 85, 212, 279, 828, 821, 829, 72, 786, 659, 782, 166, 797, 734,
  620, 66, 810, 674, 328, 197, 521, 678, 767, 714, 811, 718, 791, 793, 682,
  165, 784, 777, 274, 115, 225, 519, 720, 657, 618, 158, 709, 765, 671, 834,
  680, 836, 181, 351, 226, 432, 716, 745, 617, 518, 705, 271, 667, 252, 406,
  795, 355, 663, 587, 430, 230, 249, 377, 189, 823, 375, 825, 464, 614, 669,
  208, 743, 664, 615, 848, 408, 780, 554, 516, 494, 771, 515, 585, 552, 642,
  23, 492, 694, 703, 532, 772, 462, 229, 754, 631, 277, 146, 843, 756, 563,
  49, 288, 646, 419, 330, 143, 558, 635, 565, 728, 366, 447, 410, 50, 689,
  505, 436, 341, 22, 591, 559, 842, 603, 688, 592, 574, 805, 543, 98, 530,
  161, 601, 481, 470, 604, 101, 219, 541, 507, 337, 353, 612, 576, 723, 477,
  326, 466, 658, 68, 722, 51, 227, 421, 404, 504, 513, 35, 451, 124, 609,
  440, 800, 799, 364, 382, 510, 54, 292, 167
};

#endif  // LBDREC_FREAK_H
