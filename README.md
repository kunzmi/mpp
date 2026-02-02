# MPP - Michael Performance Primitives

Intel has [IPP](https://www.intel.com/content/www/us/en/developer/tools/oneapi/ipp.html), Nvidia has [NPP](https://docs.nvidia.com/cuda/npp/) – and I have MPP: Michael Performance Primitives (actually not that easy to find a three letter name where the last two letters are given and that is still available...). 

For a very long time now, many – if not all – of my cuda related projects rely in some way on NPP. I even wrote a C# based wrapper for this library as a part of [managedCuda](https://github.com/kunzmi/managedCuda) and all functions in NPP have a counterpart in C#: I thus know how many functions NPP provides (more than 5000 including all variants).

But from [managedCuda](https://github.com/kunzmi/managedCuda) I also know that many things are repetitive in NPP: nppiAdd does (mostly) the same for all datatypes, nppiSub is basically the same as nppiAdd, etc. If one only counts the actual functionalities independent of the data type, that means leaving out different variations, temp-buffer size and other helper functions, the number of 5000 functions quickly reduces to less than 300. A feasible number. Once I knew that number, the idea for MPP was born – because even though the number of functions is large, there is always one function missing: a special edge case, a special variant, a different datatype. At the end, one usually must implement a similar kernel with hopefully a similar performance. But to do so, what is the special ingredient that makes NPP a set of performance primitives and not just another vectorAdd-hello-world example?  

Working now with Cuda for more than 15 years, this was not per se the most difficult problem to solve – the question was more to get things generalized and scaled to all NPP-use-cases: The solution is to use C++ templates and making heavy use of concepts introduced in C++20. And a different approach to handle unaligned data as compared to NPP, but this would go beyond the scope of a project readme.md – I’ll likely fill the wiki with some information on that over time. 

Finally, now that I implemented the entire imaging subset of NPP for all datatypes, I can say that it is not that complicated to achieve the performance of NPP, it is even possible to become faster than NPP, sometimes even much faster!

I compared the average runtime of a few exemplery primitives in this table: [Performance Analysis](/performanceAnalysis/results.html). 

The short resume: 
- Aligned data is either as fast as NPP or faster (with very few exceptions).
- Unaligned data is significantly faster with MPP than with NPP due to my one-kernel-approach.
- Box- or Gauss-filtering with large kernels is (up to 90x) faster with MPP than with NPP. All tests were done on a RTX4090.
- Median-filtering with NPP is quite slow, I've seen up to 400x improvement. But for the time beeing, only kernel sizes 3x3, 5x5 and 7x7 are implemented, NPP supports all possible kernel sizes.

To date, nearly all functions of the image processing subset of NPP are implemented – except for a few functions where I either have not yet decided on how to implement them or I simply have no idea what they are good for, how to use them or the NPP/IPP description is not in agreement with the actually computed outcome. Also missing is batch-processing and the 1D signal processing part of NPP.

# What is already available in MPP:

- All functionality in 
  - arithmetic
  - colorConversion, except nppiAlphaCompColorKey, nppiLUTPaletteSwap, nppiRGBToCbYCr422Gamma
  - dataExchangeAndInit
  - geometricTransforms 
  - morphology
  - thresholdAndCompare 
  - statistics
  - filtering is also complete except for nppiHistogramOfGradients, nppiFilterGaussPyramidLayerUp/Down, nppiFilterHoughLine, nppiFloodFill, nppiLabelMarkersUF, nppiSegmentWatershed and nppiSignedDistanceTransform

- Some new functions that I always missed in NPP, e.g. conjugate complex multiplication

- Supported datatypes are: 
  - unsigned byte (**8u**)
  - signed byte (**8s**)
  - unsigned short (**16u**)
  - signed short (**16s**)
  - unsigned int (**32u**)
  - signed int (**32s**)
  - in some exceptional cases also 64bit int (signed/unsigned).
  - 16bit half-float (**16f**)
  - 16bit bfloat (**16bf**)
  - 32bit float (**32f**)
  - 64bit float (**64f**)
  - 16bit signed complex int (**16sc**)
  - 32bit signed complex int (**32sc**)
  - 32bit complex float (**32fc**)
- One (**C1**), two (**C2**), three (**C3**), four (**C4**) channel images and three channels plus alpha channel (**AC4**)
- All kernels are implemented for all meaningful datatypes (e.g. no min/max operations on complex numbers). Using templates for the different channel counts and different datatypes makes this relatively easy to realize: one just needs to add another instantiation.
- All interpolation methods are available in all kernels that use interpolation, i.e.: 
  - NearestNeighbor
  - Linear
  - CubicHermiteSpline (called ‘cubic’ e.g. in Matlab)
  - CubicLagrange (‘cubic’ in NPP)
  - Cubic2ParamBSpline
  - Cubic2ParamCatmullRom
  - Cubic2ParamB05C03
  - SuperSampling (only for image downsampling)
  - Lanczos2Lobed
  - Lanczos3Lobed
- All kernels that might need to read pixels outside the image area (e.g. filtering, geometric transformation, etc.) support border processing in a similar way to the _Border variants in NPP. Except that NPP only supports the border type “Replicate”, MPP implements: 
  - None
  - Constant
  - Replicate
  - Mirror
  - MirrorReplicate
  - Wrap
  - SmoothEdge, which mimics IPP's SmoothEdge edge feature and which is not implemented in NPP.
- A C++ class-based wrapper for NPP like the one in managedCuda – initially the idea was to use NPP as a reference implementation but I quickly gave that up: NPP has too many bugs and inconsistencies that it could be used as a reference. The code is still there though and was usefull for performance comparisons.
- All cuda kernels are also implemented in a “simpleCPU” variant using no optimization and no parallelization at all, but the same operators and functors as the cuda kernels. This simplified variant is now considered as the reference implementation, the gold standard that all computed values must reach.
- Compiles on Windows with VisualC++ and Linux with GCC 13.3+
- No dependencies other than provided in the repository (except Cuda SDK)

# The TODO list gets smaller and smaller, things that I want to add:
- There's no install, releasing or packaging, CMake as of now just compiles the code and that's it.
- Signal processing part of NPP/IPP, i.e. NPPs or IPPs?

# What is next?
- Once MPP is fully implemented with Cuda, the next logical step would be to port the cuda kernels to hip for AMD GPUs.
- Large parts of the code could also be reused for a CPU implementation using SSE or AVX, which then leads to the question: how easy is it to reach the performance of IPP on CPU? Arm’s Neon vector units would then also benefit from such an implementation…

# A full list of the currently implemented functions in MPP:

## Arithmetic and Logical

|Name|8u|8s|16u|16s|32u|32s|16f|16bf|32f|64f|16sc|32sc|32fc|\||C1|C2|C3|C4|AC4|
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
|Abs|   | ✓ |   | ✓ |   | ✓ | ✓ | ✓ | ✓ | ✓ |   |   |   |\|| ✓ | ✓ | ✓ | ✓ | ✓ |
|AbsDiff| ✓ |   | ✓ |   | ✓ |   | ✓ | ✓ | ✓ | ✓ |   |   |   |\|| ✓ | ✓ | ✓ | ✓ | ✓ |
|AbsDiffC| ✓ |   | ✓ |   | ✓ |   | ✓ | ✓ | ✓ | ✓ |   |   |   |\|| ✓ | ✓ | ✓ | ✓ | ✓ |
|AbsDiffDevC| ✓ |   | ✓ |   | ✓ |   | ✓ | ✓ | ✓ | ✓ |   |   |   |\|| ✓ | ✓ | ✓ | ✓ | ✓ |
|Add| ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |\|| ✓ | ✓ | ✓ | ✓ | ✓ |
|AddC| ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |\|| ✓ | ✓ | ✓ | ✓ | ✓ |
|AddDevC| ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |\|| ✓ | ✓ | ✓ | ✓ | ✓ |
|AddProduct| ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |\|| ✓ | ✓ | ✓ | ✓ | ✓ |
|AddSquare| ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |\|| ✓ | ✓ | ✓ | ✓ | ✓ |
|AddWeighted| ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |\|| ✓ | ✓ | ✓ | ✓ | ✓ |
|AlphaComp| ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |   |   |   |\|| ✓ | ✓ | ✓ | ✓ |   |
|AlphaCompC| ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |   |   |   |\|| ✓ | ✓ | ✓ | ✓ | ✓ |
|AlphaPremul| ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |   |   |   |\||   |   |   | ✓ |   |
|AlphaPremulC| ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |   |   |   |\|| ✓ | ✓ | ✓ | ✓ | ✓ |
|And| ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |   |   |   |   |   |   |   |\|| ✓ | ✓ | ✓ | ✓ | ✓ |
|AndC| ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |   |   |   |   |   |   |   |\|| ✓ | ✓ | ✓ | ✓ | ✓ |
|AndDevC| ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |   |   |   |   |   |   |   |\|| ✓ | ✓ | ✓ | ✓ | ✓ |
|Div| ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |\|| ✓ | ✓ | ✓ | ✓ | ✓ |
|DivC| ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |\|| ✓ | ✓ | ✓ | ✓ | ✓ |
|DivDevC| ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |\|| ✓ | ✓ | ✓ | ✓ | ✓ |
|DivInv<sup>1</sup>| ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |\|| ✓ | ✓ | ✓ | ✓ | ✓ |
|DivInvC<sup>1</sup>| ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |\|| ✓ | ✓ | ✓ | ✓ | ✓ |
|DivInvDevC<sup>1</sup>| ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |\|| ✓ | ✓ | ✓ | ✓ | ✓ |
|Exp| ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |\|| ✓ | ✓ | ✓ | ✓ | ✓ |
|LShiftC| ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |   |   |   |   |   |   |   |\|| ✓ | ✓ | ✓ | ✓ | ✓ |
|Ln| ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |\|| ✓ | ✓ | ✓ | ✓ | ✓ |
|Mul| ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |\|| ✓ | ✓ | ✓ | ✓ | ✓ |
|MulC| ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |\|| ✓ | ✓ | ✓ | ✓ | ✓ |
|MulDevC| ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |\|| ✓ | ✓ | ✓ | ✓ | ✓ |
|MulScale| ✓ |   | ✓ |   |   |   |   |   |   |   |   |   |   |\|| ✓ | ✓ | ✓ | ✓ | ✓ |
|MulScaleC| ✓ |   | ✓ |   |   |   |   |   |   |   |   |   |   |\|| ✓ | ✓ | ✓ | ✓ | ✓ |
|MulScaleDevC| ✓ |   | ✓ |   |   |   |   |   |   |   |   |   |   |\|| ✓ | ✓ | ✓ | ✓ | ✓ |
|Not| ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |   |   |   |   |   |   |   |\|| ✓ | ✓ | ✓ | ✓ | ✓ |
|Or| ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |   |   |   |   |   |   |   |\|| ✓ | ✓ | ✓ | ✓ | ✓ |
|OrC| ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |   |   |   |   |   |   |   |\|| ✓ | ✓ | ✓ | ✓ | ✓ |
|OrDevC| ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |   |   |   |   |   |   |   |\|| ✓ | ✓ | ✓ | ✓ | ✓ |
|RShiftC| ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |   |   |   |   |   |   |   |\|| ✓ | ✓ | ✓ | ✓ | ✓ |
|Sqr| ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |\|| ✓ | ✓ | ✓ | ✓ | ✓ |
|Sqrt| ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |\|| ✓ | ✓ | ✓ | ✓ | ✓ |
|Sub| ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |\|| ✓ | ✓ | ✓ | ✓ | ✓ |
|SubC| ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |\|| ✓ | ✓ | ✓ | ✓ | ✓ |
|SubDevC| ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |\|| ✓ | ✓ | ✓ | ✓ | ✓ |
|SubInv<sup>1</sup>| ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |\|| ✓ | ✓ | ✓ | ✓ | ✓ |
|SubInvC<sup>1</sup>| ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |\|| ✓ | ✓ | ✓ | ✓ | ✓ |
|SubInvDevC<sup>1</sup>| ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |\|| ✓ | ✓ | ✓ | ✓ | ✓ |
|Xor| ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |   |   |   |   |   |   |   |\|| ✓ | ✓ | ✓ | ✓ | ✓ |
|XorC| ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |   |   |   |   |   |   |   |\|| ✓ | ✓ | ✓ | ✓ | ✓ |
|XorDevC| ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |   |   |   |   |   |   |   |\|| ✓ | ✓ | ✓ | ✓ | ✓ |

<sup>1</sup>I added this variant to allow for inplace operation `SrcDst = Src2 - SrcDst` or `SrcDst = Src2 / SrcDst` in addition to `SrcDst = SrcDst - Src2`

## Color Conversion

|Name|8u|8s|16u|16s|32u|32s|16f|16bf|32f|64f|16sc|32sc|32fc|\||C1|C2|C3|C4|AC4|
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
|BGRtoHLS| ✓ |   | ✓ |   |   |   | ✓ |   | ✓ |   |   |   |   |\||   |   | ✓ | ✓ | ✓ |
|BGRtoHSV| ✓ |   | ✓ |   |   |   | ✓ |   | ✓ |   |   |   |   |\||   |   | ✓ | ✓ | ✓ |
|BGRtoLUV| ✓ |   | ✓ |   |   |   | ✓ |   | ✓ |   |   |   |   |\||   |   | ✓ | ✓ | ✓ |
|BGRtoLab| ✓ |   | ✓ |   |   |   | ✓ |   | ✓ |   |   |   |   |\||   |   | ✓ | ✓ | ✓ |
|CFAToRGB| ✓ |   | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |   |   |   |   |\|| ✓ |   |   |   |   |
|ColorToGray| ✓ |   | ✓ | ✓ |   |   | ✓ |   | ✓ |   |   |   |   |\||   | ✓ | ✓ | ✓ | ✓ |
|ColorTwist3x3| ✓ |   | ✓ | ✓ |   |   | ✓ |   | ✓ |   |   |   |   |\||   |   | ✓ | ✓ | ✓ |
|ColorTwist3x3From411| ✓ |   | ✓ | ✓ |   |   | ✓ |   | ✓ |   |   |   |   |\||   |   | ✓ |   | ✓ |
|ColorTwist3x3From420| ✓ |   | ✓ | ✓ |   |   | ✓ |   | ✓ |   |   |   |   |\||   |   | ✓ |   | ✓ |
|ColorTwist3x3From422| ✓ |   | ✓ | ✓ |   |   | ✓ |   | ✓ |   |   |   |   |\||   | ✓ | ✓ |   | ✓ |
|ColorTwist3x3To411| ✓ |   | ✓ | ✓ |   |   | ✓ |   | ✓ |   |   |   |   |\||   |   | ✓ |   | ✓ |
|ColorTwist3x3To420| ✓ |   | ✓ | ✓ |   |   | ✓ |   | ✓ |   |   |   |   |\||   |   | ✓ |   | ✓ |
|ColorTwist3x3To422| ✓ |   | ✓ | ✓ |   |   | ✓ |   | ✓ |   |   |   |   |\||   |   | ✓ |   | ✓ |
|ColorTwist3x4<sup>1</sup>| ✓ |   | ✓ | ✓ |   |   | ✓ |   | ✓ |   |   |   |   |\||   |   | ✓ | ✓ | ✓ |
|ColorTwist3x4From411<sup>1</sup>| ✓ |   | ✓ | ✓ |   |   | ✓ |   | ✓ |   |   |   |   |\||   |   | ✓ |   | ✓ |
|ColorTwist3x4From420<sup>1</sup>| ✓ |   | ✓ | ✓ |   |   | ✓ |   | ✓ |   |   |   |   |\||   |   | ✓ |   | ✓ |
|ColorTwist3x4From422<sup>1</sup>| ✓ |   | ✓ | ✓ |   |   | ✓ |   | ✓ |   |   |   |   |\||   | ✓ | ✓ |   | ✓ |
|ColorTwist3x4To411<sup>1</sup>| ✓ |   | ✓ | ✓ |   |   | ✓ |   | ✓ |   |   |   |   |\||   |   | ✓ |   | ✓ |
|ColorTwist3x4To420<sup>1</sup>| ✓ |   | ✓ | ✓ |   |   | ✓ |   | ✓ |   |   |   |   |\||   |   | ✓ |   | ✓ |
|ColorTwist3x4To422<sup>1</sup>| ✓ |   | ✓ | ✓ |   |   | ✓ |   | ✓ |   |   |   |   |\||   |   | ✓ |   | ✓ |
|ColorTwist4x4| ✓ |   | ✓ | ✓ |   |   | ✓ |   | ✓ |   |   |   |   |\||   |   |   | ✓ |   |
|ColorTwist4x4C| ✓ |   | ✓ | ✓ |   |   | ✓ |   | ✓ |   |   |   |   |\||   |   |   | ✓ |   |
|CompColorKey| ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |   |   |   |\|| ✓ | ✓ | ✓ | ✓ | ✓ |
|ConvertSampling422<sup>2</sup>| ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |   |   |   |\||   | ✓ | ✓ |   |   |
|GammaCorrBT709<sup>3</sup>| ✓ |   | ✓ | ✓ |   |   | ✓ |   | ✓ |   |   |   |   |\|| ✓ | ✓ | ✓ | ✓ | ✓ |
|GammaCorrsRGB<sup>3</sup>| ✓ |   | ✓ | ✓ |   |   | ✓ |   | ✓ |   |   |   |   |\|| ✓ | ✓ | ✓ | ✓ | ✓ |
|GammaInvCorrBT709<sup>3</sup>| ✓ |   | ✓ | ✓ |   |   | ✓ |   | ✓ |   |   |   |   |\|| ✓ | ✓ | ✓ | ✓ | ✓ |
|GammaInvCorrsRGB<sup>3</sup>| ✓ |   | ✓ | ✓ |   |   | ✓ |   | ✓ |   |   |   |   |\|| ✓ | ✓ | ✓ | ✓ | ✓ |
|GradientColorToGray| ✓ |   | ✓ | ✓ |   |   | ✓ |   | ✓ |   |   |   |   |\||   | ✓ | ✓ | ✓ | ✓ |
|HLStoBGR| ✓ |   | ✓ |   |   |   | ✓ |   | ✓ |   |   |   |   |\||   |   | ✓ | ✓ | ✓ |
|HLStoRGB| ✓ |   | ✓ |   |   |   | ✓ |   | ✓ |   |   |   |   |\||   |   | ✓ | ✓ | ✓ |
|HSVtoBGR| ✓ |   | ✓ |   |   |   | ✓ |   | ✓ |   |   |   |   |\||   |   | ✓ | ✓ | ✓ |
|HSVtoRGB| ✓ |   | ✓ |   |   |   | ✓ |   | ✓ |   |   |   |   |\||   |   | ✓ | ✓ | ✓ |
|LUT<sup>4</sup>|   |   |   |   |   |   | ✓ | ✓ | ✓ |   |   |   |   |\|| ✓ | ✓ | ✓ | ✓ | ✓ |
|LUTPalette| ✓ |   | ✓ | ✓ |   |   |   |   |   |   |   |   |   |\|| ✓ | ✓ | ✓ | ✓ | ✓ |
|LUTPaletteAC4| ✓ |   | ✓ | ✓ |   |   |   |   |   |   |   |   |   |\|| ✓ |   |   |   |   |
|LUTPaletteC3| ✓ |   | ✓ | ✓ |   |   |   |   |   |   |   |   |   |\|| ✓ |   |   |   |   |
|LUTTrilinear| ✓ |   | ✓ | ✓ |   |   |   |   | ✓ |   |   |   |   |\||   |   | ✓ | ✓ | ✓ |
|LUTTrilinearAC4| ✓ |   | ✓ | ✓ |   |   |   |   | ✓ |   |   |   |   |\||   |   | ✓ | ✓ | ✓ |
|LUVtoBGR| ✓ |   | ✓ |   |   |   | ✓ |   | ✓ |   |   |   |   |\||   |   | ✓ | ✓ | ✓ |
|LUVtoRGB| ✓ |   | ✓ |   |   |   | ✓ |   | ✓ |   |   |   |   |\||   |   | ✓ | ✓ | ✓ |
|LabtoBGR| ✓ |   | ✓ |   |   |   | ✓ |   | ✓ |   |   |   |   |\||   |   | ✓ | ✓ | ✓ |
|LabtoRGB| ✓ |   | ✓ |   |   |   | ✓ |   | ✓ |   |   |   |   |\||   |   | ✓ | ✓ | ✓ |
|RGBToCFA<sup>5</sup>| ✓ |   | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |   |   |   |   |\||   |   | ✓ |   | ✓ |
|RGBtoHLS| ✓ |   | ✓ |   |   |   | ✓ |   | ✓ |   |   |   |   |\||   |   | ✓ | ✓ | ✓ |
|RGBtoHSV| ✓ |   | ✓ |   |   |   | ✓ |   | ✓ |   |   |   |   |\||   |   | ✓ | ✓ | ✓ |
|RGBtoLUV| ✓ |   | ✓ |   |   |   | ✓ |   | ✓ |   |   |   |   |\||   |   | ✓ | ✓ | ✓ |
|RGBtoLab| ✓ |   | ✓ |   |   |   | ✓ |   | ✓ |   |   |   |   |\||   |   | ✓ | ✓ | ✓ |

<sup>1</sup>The NPP API provides individual functions for different color space conversions, such as RGBtoYCbCr. But then variants exist for planar configurations in RGBtoYCbCr but not for BGRtoYCbCr and vice versa - as internally all of these conversions can be represented by a 3x4 matrix operation, I opted for a gereneralized API and provide the user the matrix values to pass to the ColorTwist3x4 function.

<sup>2</sup>MPP only provides a primitive to convert from and to 422 C2 packed format to/from 2/3-channel planar. All other resampling operations on planar data can be replaced by either a Copy-function or a Resize-function.

<sup>3</sup>The gamma curve that NPP and IPP are using corresponds to the 'BT709' gamma curve, but most common is likely the sRGB curve, so MPP provides both.

<sup>4</sup>For 8u and 16u data, it is more performant to pre-compute all possible LUT values in a palette and then use the LUTPalette function. MPP provides routines to compute a palette from a LUT on host and on device.

<sup>5</sup>The inverse of CFAtoRGB seems like a nice to have


## Data Exchange and Initialization

|Name|8u|8s|16u|16s|32u|32s|16f|16bf|32f|64f|16sc|32sc|32fc|\||C1|C2|C3|C4|AC4|
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
|Copy| ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |\|| ✓ | ✓ | ✓ | ✓ | ✓ |
|CopyBorder| ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |\|| ✓ | ✓ | ✓ | ✓ | ✓ |
|CopySubpix| ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |\|| ✓ | ✓ | ✓ | ✓ | ✓ |
|SetC| ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |\|| ✓ | ✓ | ✓ | ✓ | ✓ |
|SetDevC| ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |\|| ✓ | ✓ | ✓ | ✓ | ✓ |
|SwapChannel| ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |\||   | ✓ | ✓ | ✓ |   |
|Transpose| ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |\|| ✓ | ✓ | ✓ | ✓ |   |


## Filtering

|Name|8u|8s|16u|16s|32u|32s|16f|16bf|32f|64f|16sc|32sc|32fc|\||C1|C2|C3|C4|AC4|
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
|BilateralGaussFilter| ✓ | ✓ | ✓ | ✓ |   |   | ✓ | ✓ | ✓ |   |   |   |   |\|| ✓ | ✓ | ✓ | ✓ | ✓ |
|BoxAndSumSquareFilter<sup>1</sup>| ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |   |   |   |   |\|| ✓ |   |   |   |   |
|BoxFilter| ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |\|| ✓ | ✓ | ✓ | ✓ | ✓ |
|CannyEdge|   |   |   | ✓ |   |   |   |   | ✓ |   |   |   |   |\|| ✓ |   |   |   |   |
|ColumnFilter| ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |\|| ✓ | ✓ | ✓ | ✓ | ✓ |
|ColumnWindowSum| ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |\|| ✓ | ✓ | ✓ | ✓ | ✓ |
|Filter| ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |\|| ✓ | ✓ | ✓ | ✓ | ✓ |
|FixedFilter<sup>2</sup>| ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |\|| ✓ | ✓ | ✓ | ✓ | ✓ |
|GradientVectorPrewitt| ✓ | ✓ | ✓ | ✓ |   |   |   |   | ✓ |   |   |   |   |\|| ✓ | ✓ | ✓ | ✓ | ✓ |
|GradientVectorScharr| ✓ | ✓ | ✓ | ✓ |   |   |   |   | ✓ |   |   |   |   |\|| ✓ | ✓ | ✓ | ✓ | ✓ |
|GradientVectorSobel| ✓ | ✓ | ✓ | ✓ |   |   |   |   | ✓ |   |   |   |   |\|| ✓ | ✓ | ✓ | ✓ | ✓ |
|HarrisCornerResponse|   |   |   |   |   |   |   |   | ✓ |   |   |   |   |\||   |   |   | ✓ |   |
|MaxFilter| ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |   |   |   |\|| ✓ | ✓ | ✓ | ✓ | ✓ |
|MedianFilter<sup>3</sup>| ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |   |   |   |\|| ✓ | ✓ | ✓ | ✓ | ✓ |
|MinFilter| ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |   |   |   |\|| ✓ | ✓ | ✓ | ✓ | ✓ |
|RowFilter| ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |\|| ✓ | ✓ | ✓ | ✓ | ✓ |
|RowWindowSum| ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |\|| ✓ | ✓ | ✓ | ✓ | ✓ |
|SeparableFilter<sup>4</sup>| ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |\|| ✓ | ✓ | ✓ | ✓ | ✓ |
|ThresholdAdaptiveBoxFilter| ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |   |   |   |\|| ✓ | ✓ | ✓ | ✓ | ✓ |
|UnsharpFilter| ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |   |   |   |\|| ✓ | ✓ | ✓ | ✓ | ✓ |
|WienerFilter| ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |   |   |   |\|| ✓ | ✓ | ✓ | ✓ | ✓ |

<sup>1</sup>A specialised box filter for one-channel images that returns in first channel result image the mean value under the box area and in the second channel the summed squared pixel values. The result can then be used as a pre-computed step in the CrossCorrelationCoefficient function.

<sup>2</sup>One function for the following filter kernels: Gauss, HighPass, LowPass, Laplace, PrewittHoriz, PrewittVert, RobertsDown, RobertsUp, ScharrHoriz, ScharrVert, Sharpen, SobelCross, SobelHoriz, SobelVert, SobelHorizSecond, SobelVertSecond

<sup>3</sup>Only for filter sizes 3x3, 5x5 and 7x7

<sup>4</sup>Replaces nppiFilterGaussAdvancedBorder

## Geometry Transforms

|Name|8u|8s|16u|16s|32u|32s|16f|16bf|32f|64f|16sc|32sc|32fc|\||C1|C2|C3|C4|AC4|
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
|Mirror| ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |\|| ✓ | ✓ | ✓ | ✓ | ✓ |
|Remap| ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |\|| ✓ | ✓ | ✓ | ✓ | ✓ |
|RemapC2<sup>1</sup>| ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |\|| ✓ | ✓ | ✓ | ✓ | ✓ |
|Resize| ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |\|| ✓ | ✓ | ✓ | ✓ | ✓ |
|ResizeSqrPixel| ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |\|| ✓ | ✓ | ✓ | ✓ | ✓ |
|Rotate| ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |\|| ✓ | ✓ | ✓ | ✓ | ✓ |
|WarpAffine| ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |\|| ✓ | ✓ | ✓ | ✓ | ✓ |
|WarpAffineBack| ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |\|| ✓ | ✓ | ✓ | ✓ | ✓ |
|WarpPerspective| ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |\|| ✓ | ✓ | ✓ | ✓ | ✓ |
|WarpPerspectiveBack| ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |\|| ✓ | ✓ | ✓ | ✓ | ✓ |

<sup>1</sup>NPP has a Remap function that takes two one-channel maps for X and Y pixel indices. MPP additionally has the same routine with a packed two-channel index-map, i.e. X and Y component in one image.

## Complex Numbers<sup>1</sup>

|Name|8u|8s|16u|16s|32u|32s|16f|16bf|32f|64f|16sc|32sc|32fc|\||C1|C2|C3|C4|AC4|
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
|Angle|   |   |   |   |   |   |   |   |   |   |   |   | ✓ |\|| ✓ | ✓ | ✓ | ✓ |   |
|Conj|   |   |   |   |   |   |   |   |   |   | ✓ | ✓ | ✓ |\|| ✓ | ✓ | ✓ | ✓ |   |
|ConjMul|   |   |   |   |   |   |   |   |   |   | ✓ | ✓ | ✓ |\|| ✓ | ✓ | ✓ | ✓ |   |
|Imag|   |   |   |   |   |   |   |   |   |   | ✓ | ✓ | ✓ |\|| ✓ | ✓ | ✓ | ✓ |   |
|Magnitude<sup>2</sup>|   |   |   |   |   |   |   |   |   |   |   |   | ✓ |\|| ✓ | ✓ | ✓ | ✓ |   |
|MagnitudeLog<sup>2</sup>|   |   |   |   |   |   |   |   |   |   |   |   | ✓ |\|| ✓ | ✓ | ✓ | ✓ |   |
|MagnitudeSqr<sup>2</sup>|   |   |   |   |   |   |   |   |   |   |   |   | ✓ |\|| ✓ | ✓ | ✓ | ✓ |   |
|MakeComplex|   |   |   | ✓ |   | ✓ |   |   | ✓ |   |   |   |   |\|| ✓ | ✓ | ✓ | ✓ |   |
|MakeComplexImag|   |   |   | ✓ |   | ✓ |   |   | ✓ |   |   |   |   |\|| ✓ | ✓ | ✓ | ✓ |   |
|Real|   |   |   |   |   |   |   |   |   |   | ✓ | ✓ | ✓ |\|| ✓ | ✓ | ✓ | ✓ |   |

<sup>1</sup>NPP offers only rudimentary support for complex numbers in a category named `linear transforms`, but geometry and color transforms are also linear transforms, why I renamed the category `Complex Numbers`.

<sup>2</sup>As a special feature, the Magnitude functions can return a full spectrum with zero frequency in the center if the input is of size `(width/2)+1`, the size of a FFTW/CUFFT R2C FFT.

## Morphological

|Name|8u|8s|16u|16s|32u|32s|16f|16bf|32f|64f|16sc|32sc|32fc|\||C1|C2|C3|C4|AC4|
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
|BlackHat| ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |   |   |   |\|| ✓ | ✓ | ✓ | ✓ | ✓ |
|Close| ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |   |   |   |\|| ✓ | ✓ | ✓ | ✓ | ✓ |
|Dilation| ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |   |   |   |\|| ✓ | ✓ | ✓ | ✓ | ✓ |
|DilationGray| ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |   |   |   |\|| ✓ | ✓ | ✓ | ✓ | ✓ |
|DilationMask| ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |   |   |   |\|| ✓ | ✓ | ✓ | ✓ | ✓ |
|Erosion| ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |   |   |   |\|| ✓ | ✓ | ✓ | ✓ | ✓ |
|ErosionGray| ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |   |   |   |\|| ✓ | ✓ | ✓ | ✓ | ✓ |
|ErosionMask| ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |   |   |   |\|| ✓ | ✓ | ✓ | ✓ | ✓ |
|MorphologyGradient<sup>1</sup>| ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |   |   |   |\|| ✓ | ✓ | ✓ | ✓ | ✓ |
|Open| ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |   |   |   |\|| ✓ | ✓ | ✓ | ✓ | ✓ |
|TopHat| ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |   |   |   |\|| ✓ | ✓ | ✓ | ✓ | ✓ |

<sup>1</sup>When one calls the `MorphGradient` functions in NPP, it returns the result of `BlackHat`

## Statistics

|Name|8u|8s|16u|16s|32u|32s|16f|16bf|32f|64f|16sc|32sc|32fc|\||C1|C2|C3|C4|AC4|
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
|AverageError| ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |\|| ✓ | ✓ | ✓ | ✓ | ✓ |
|AverageRelativeError| ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |\|| ✓ | ✓ | ✓ | ✓ | ✓ |
|CircularRadialProfile| ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |   |   |   |   |\|| ✓ | ✓ | ✓ | ✓ | ✓ |
|CountInRange| ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |   |   |   |\|| ✓ | ✓ | ✓ | ✓ | ✓ |
|CrossCorrelation| ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |   |   |   |   |\|| ✓ |   |   |   |   |
|CrossCorrelationCoefficient| ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |   |   |   |   |\|| ✓ |   |   |   |   |
|CrossCorrelationNormalized| ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |   |   |   |   |\|| ✓ |   |   |   |   |
|DotProduct| ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |\|| ✓ | ✓ | ✓ | ✓ | ✓ |
|EllipticalRadialProfile<sup>1</sup>| ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |   |   |   |   |\|| ✓ | ✓ | ✓ | ✓ | ✓ |
|HistogramEven| ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |   |   |   |\|| ✓ | ✓ | ✓ | ✓ | ✓ |
|HistogramRange| ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |   |   |   |\|| ✓ | ✓ | ✓ | ✓ | ✓ |
|Integral| ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |   |   |   |\|| ✓ | ✓ | ✓ | ✓ |   |
|MSE| ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |\|| ✓ | ✓ | ✓ | ✓ | ✓ |
|MSSSIM| ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |   |   |   |\|| ✓ | ✓ | ✓ | ✓ | ✓ |
|Max| ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |   |   |   |\|| ✓ | ✓ | ✓ | ✓ | ✓ |
|MaxEvery| ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |   |   |   |\|| ✓ | ✓ | ✓ | ✓ | ✓ |
|MaxIndex| ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |   |   |   |\|| ✓ | ✓ | ✓ | ✓ | ✓ |
|MaximumError| ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |\|| ✓ | ✓ | ✓ | ✓ | ✓ |
|MaximumRelativeError| ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |\|| ✓ | ✓ | ✓ | ✓ | ✓ |
|Mean| ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |\|| ✓ | ✓ | ✓ | ✓ | ✓ |
|MeanStd| ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |\|| ✓ | ✓ | ✓ | ✓ | ✓ |
|Min| ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |   |   |   |\|| ✓ | ✓ | ✓ | ✓ | ✓ |
|MinEvery| ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |   |   |   |\|| ✓ | ✓ | ✓ | ✓ | ✓ |
|MinIndex| ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |   |   |   |\|| ✓ | ✓ | ✓ | ✓ | ✓ |
|MinMax| ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |   |   |   |\|| ✓ | ✓ | ✓ | ✓ | ✓ |
|MinMaxIndex| ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |   |   |   |\|| ✓ | ✓ | ✓ | ✓ | ✓ |
|NormDiffInf| ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |   |   |   |\|| ✓ | ✓ | ✓ | ✓ | ✓ |
|NormDiffL1| ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |   |   |   |\|| ✓ | ✓ | ✓ | ✓ | ✓ |
|NormDiffL2| ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |   |   |   |\|| ✓ | ✓ | ✓ | ✓ | ✓ |
|NormInf| ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |   |   |   |\|| ✓ | ✓ | ✓ | ✓ | ✓ |
|NormL1| ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |   |   |   |\|| ✓ | ✓ | ✓ | ✓ | ✓ |
|NormL2| ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |   |   |   |\|| ✓ | ✓ | ✓ | ✓ | ✓ |
|NormRelInf| ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |   |   |   |\|| ✓ | ✓ | ✓ | ✓ | ✓ |
|NormRelL1| ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |   |   |   |\|| ✓ | ✓ | ✓ | ✓ | ✓ |
|NormRelL2| ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |   |   |   |\|| ✓ | ✓ | ✓ | ✓ | ✓ |
|PSNR| ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |   |   |   |\|| ✓ | ✓ | ✓ | ✓ | ✓ |
|QualityIndex<sup>2</sup>| ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |   |   |   |\|| ✓ | ✓ | ✓ | ✓ | ✓ |
|QualityIndexWindow<sup>2</sup>| ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |   |   |   |\|| ✓ | ✓ | ✓ | ✓ | ✓ |
|RectStdDev|   |   |   |   |   | ✓ |   |   | ✓ | ✓ |   |   |   |\|| ✓ | ✓ | ✓ | ✓ |   |
|SSIM| ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |   |   |   |\|| ✓ | ✓ | ✓ | ✓ | ✓ |
|SqrIntegral| ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |   |   |   |\|| ✓ | ✓ | ✓ | ✓ |   |
|SquareDistanceNormalized| ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |   |   |   |   |\|| ✓ |   |   |   |   |
|Sum| ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |\|| ✓ | ✓ | ✓ | ✓ | ✓ |

<sup>1</sup>I never managed to get a result out of EllipticalRadialProfile in NPP.

<sup>2</sup>QualityIndex is identical to the one in NPP computing a global index without a sliding window. QualityIndexWindow is implemented using a sliding window approach as is done in the original paper / code with a window size of 11x11 pixels.

## Threshold and Sompare

|Name|8u|8s|16u|16s|32u|32s|16f|16bf|32f|64f|16sc|32sc|32fc|\||C1|C2|C3|C4|AC4|
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
|Compare<sup>1</sup>| ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |\|| ✓ | ✓ | ✓ | ✓ | ✓ |
|CompareC<sup>1</sup>| ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |\|| ✓ | ✓ | ✓ | ✓ | ✓ |
|CompareDevC<sup>1</sup>| ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |\|| ✓ | ✓ | ✓ | ✓ | ✓ |
|CompareEqEps|   |   |   |   |   |   | ✓ | ✓ | ✓ | ✓ |   |   | ✓ |\|| ✓ | ✓ | ✓ | ✓ | ✓ |
|CompareEqEpsC|   |   |   |   |   |   | ✓ | ✓ | ✓ | ✓ |   |   | ✓ |\|| ✓ | ✓ | ✓ | ✓ | ✓ |
|CompareEqEpsDevC|   |   |   |   |   |   | ✓ | ✓ | ✓ | ✓ |   |   | ✓ |\|| ✓ | ✓ | ✓ | ✓ | ✓ |
|CompareFloat<sup>2</sup>|   |   |   |   |   |   | ✓ | ✓ | ✓ | ✓ |   |   | ✓ |\|| ✓ | ✓ | ✓ | ✓ | ✓ |
|ReplaceIf<sup>3</sup>| ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |\|| ✓ | ✓ | ✓ | ✓ | ✓ |
|ReplaceIfC<sup>3</sup>| ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |\|| ✓ | ✓ | ✓ | ✓ | ✓ |
|ReplaceIfDevC<sup>3</sup>| ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |\|| ✓ | ✓ | ✓ | ✓ | ✓ |
|ReplaceIfFloat<sup>3</sup>|   |   |   |   |   |   | ✓ | ✓ | ✓ | ✓ |   |   | ✓ |\|| ✓ | ✓ | ✓ | ✓ | ✓ |
|Threshold| ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |   |   |   |\|| ✓ | ✓ | ✓ | ✓ | ✓ |
|ThresholdDevC| ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |   |   |   |\|| ✓ | ✓ | ✓ | ✓ | ✓ |
|ThresholdGT| ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |   |   |   |\|| ✓ | ✓ | ✓ | ✓ | ✓ |
|ThresholdGTDevC| ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |   |   |   |\|| ✓ | ✓ | ✓ | ✓ | ✓ |
|ThresholdGTVal| ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |   |   |   |\|| ✓ | ✓ | ✓ | ✓ | ✓ |
|ThresholdLT| ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |   |   |   |\|| ✓ | ✓ | ✓ | ✓ | ✓ |
|ThresholdLTDevC| ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |   |   |   |\|| ✓ | ✓ | ✓ | ✓ | ✓ |
|ThresholdLTGT| ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |   |   |   |\|| ✓ | ✓ | ✓ | ✓ | ✓ |
|ThresholdLTVal| ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |   |   |   |\|| ✓ | ✓ | ✓ | ✓ | ✓ |
|ThresholdVal| ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |   |   |   |\|| ✓ | ✓ | ✓ | ✓ | ✓ |

<sup>1</sup>Additionally to what NPP allows, MPP can apply the compare-operation per channel or for all channels and return a result for each channel or for all channels together.

<sup>2</sup>CompareFloat allows to check for unnormalized floating values, such as INF or NaN, a feature I always missed in NPP.

<sup>3</sup>ReplaceIf, replaces a pixel value if the compare-operation returns true: Basically a combined Compare/Set-with-mask operation.