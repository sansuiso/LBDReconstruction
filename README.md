LBDReconstruction
=================

This is the companion code for our work on Local Binary Descriptors (LBD) reconstruction:

* From Bits to Images: Inversion of Local Binary Descriptors (submitted) [ \[pre-print\] ][arxiv]
* Beyond Bits: Reconstructing Images from Local Binary Descriptors, ICPR 2012 [ \[pre-print\] ][icpr12]

It was used to generate the figures in the above references.

### Authors

* [Emmanuel d'Angelo](https://www.linkedin.com/in/emmanueldangelo/) (code, contact author)
* [Alexandre Alahi](https://people.epfl.ch/alexandre.alahi)
* [Laurent Jacques](https://perso.uclouvain.be/laurent.jacques/)
* [Pierre Vandergheynst](https://people.epfl.ch/pierre.vandergheynst)

## Installing the code

### Supported platforms

The software is written in C++ (with C++14 set as the required standard). Installing and using it should be straightforward for any platform where the dependencies are satisfied.
Note however that it was developed and tested on macOS only.

### Dependencies

* Cmake >= 3.13, <http://www.cmake.org>
* OpenCV >= 4.0, <http://opencv.org>
* (optional) Doxygen, <http://www.doxygen.org>
* (optional) libdispatch (a.k.a. Grand Central Dispatch on MacOS)

Optionally, the code can take advantage of [Grand Central Dispatch](http://en.wikipedia.org/wiki/Grand_central_dispatch) on platforms that provide an implementation of libdispatch (MacOS, Linux, FreeBSD...).

### Building 

You should respect the Cmake approach of out-of-source building for better results.
When using Cmake from the command line, the following specific options are available:

* WITH\_DISPATCH (ON/OFF, default: OFF) : use Grand Central Dispatch for parallel execution
* BUILD\_DOC (ON/OFF, default. OFF) : build the documentation with Doxygen (not much to build right now...)

## Code summary

The code implements the following reconstruction algorithms:

* non-binary, Total Variation + L1 error (TV-L1) reconstruction, primal-dual solver used in [this paper][icpr12]
* non-binary, Wavelet sparsity + L1 error (W-L1) reconstruction, primal-dual solver presented in [Section 3.1  of this paper][arxiv]
* binary, wavelet sparsity + binary consistency reconstruction, Binary Iterative Hard Thresholding (BIHT) solver presented in [Section 3.2][arxiv]

The Local Binary Descriptors were re-written from scratch in order to provide more flexibility and to fit in the optimization framework.
The following LBDs are implemented:

* [Binary Robust Independent Elementary Feature][brief] (BRIEF) in its uniform distribution variant
* [Fast Retina Keypoint][freak] (FREAK)
* Random FREAK: similar to FREAK with random selection of the difference pairs instead
* Exhaustive FREAK: similar to FREAK but uses all the possible difference pairs instead.

New LBDs can be added by subclassing the abstract class `LBDOperator`.

### Code organization

The layout of the code is as follows:

src  
 |  
 |\-\- LBDReconstruction : small static library implementing LBDs and reconstruction algorithms  
 |  
 |\-\- utils : various command line tools to plot, draw, compute constants...  
 |  
 |\-\- real : command line tools to apply the TV-L1 and W-L1 algorithms above  
 |  
 |\-\- binary : command line tools to apply the BIHT algorithm above  

### Naming patterns

If the name of a file starts with an `I`, it declares or implements an abstract class.

* Example : the file `ILinearOperator.hpp` declares the abstract class `LinearOperator`.

All the classes are embedded in an `lts2` namespace.

## TODO

* Improve Doxygen documentation
* Unify files/functions/classes naming patterns
* More flexible optimization (math) functions using C++11
* Replace Grand Central Dispatch with OpenCV's new parallel_for()
* Faster wavelet transform (using a parallel algorithm)

## Acknowledgement

The wavelet transform was initially implemented in C by [Jérôme Gilles](http://www.math.ucla.edu/~jegilles/) from the classical book of Stéphane Mallat [A Wavelet Tour of Signal Processing](http://www.amazon.com/exec/obidos/tg/detail/-/012466606X/).

## References

_From Bits to Images: Inversion of Local Binary Descriptors_ [ \[pre-print\] ][arxiv]  
E. d'Angelo, L. Jacques, A. Alahi, P. Vandergheynst  
IEEE Transactions on Pattern Analysis and Machine Intelligence (TPAMI) 36 (5), 874-887, 2013.

_Beyond Bits: Reconstructing Images from Local Binary Descriptors_ [ \[pre-print\] ][icpr12]  
E. d'Angelo, A. Alahi, P. Vandergheynst  
21st International Conference on Pattern Recognition (ICPR), 2012.

_FREAK: Fast Retina Keypoint_ [ \[pre-print\] ][freak]  
A. Alahi, R. Ortiz, P. Vandergheynst  
IEEE Conference on Computer Vision and Pattern Recognition (CVPR), 2012.  

_Binary Robust Independent Elementary Features_ [ \[homepage\] ][brief]  
M. Calonder, V. Lepetit, C. Strecha, P. Fua
11th European Conference on Computer Vision (ECCV), 2010.

[icpr12]: http://infoscience.epfl.ch/record/178299 "Beyond Bits: Reconstructing Images from Local Binary Descriptors"

[arxiv]: https://arxiv.org/abs/1211.1265 "From Bits to Images: Inversion of Local Binary Descriptors"

[brief]: http://cvlab.epfl.ch/research/detect/brief/ "BRIEF homepage"

[freak]: http://infoscience.epfl.ch/record/175537 "FREAK"
