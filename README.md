# MPP - Michael Performance Primitives

Intel has [IPP](https://www.intel.com/content/www/us/en/developer/tools/oneapi/ipp.html), Nvidia has [NPP](https://docs.nvidia.com/cuda/npp/) – and I have MPP: Michael Performance Primitives (actually not that easy to find a three letter name where the last two letters are given and that is still available...). 

For a very long time now, many – if not all – of my cuda related projects rely in some way on NPP. I even wrote a C# based wrapper for this library as a part of [managedCuda](https://github.com/kunzmi/managedCuda) and all functions in NPP have a counterpart in C#: I thus know how many functions NPP provides (more than 5000 including all variants).

But from [managedCuda](https://github.com/kunzmi/managedCuda) I also know that many things are repetitive in NPP: nppiAdd does (mostly) the same for all datatypes, nppiSub is basically the same as nppiAdd, etc. If one only counts the actual functionalities independent of the data type, that means leaving out different variations, temp-buffer size and other helper functions, the number of 5000 functions quickly reduces to less than 300. A feasible number. Once I knew that number, the idea for MPP was born – because even though the number of functions is large, there is always one function missing: a special edge case, a special variant, a different datatype. At the end, one usually must implement a similar kernel with hopefully a similar performance. But to do so, what is the special ingredient that makes NPP a set of performance primitives and not just another vectorAdd-hello-world example?  

 Working now with Cuda for more than 15 years, this was not per se the most difficult problem to solve – the question was more to get things generalized and scaled to all NPP-use-cases: The solution is to use C++ templates and making heavy use of concepts introduced in C++20. And a different approach to handle unaligned data compared to NPP, but this would go beyond the scope of a project readme.md – I’ll likely fill the wiki with some information on that over time. 

Finally, now that I implemented a very large subset of NPP for all datatypes, I can say that it is not that complicated to achieve the performance of NPP, it is even possible to become faster than NPP, sometimes even much faster!

I compared the average runtime of a few exemplery primitives in this table: [Performance Analysis](/performanceAnalysis/results.html). 

The short resume: 
- Aligned data is either as fast as NPP or faster (with a few exceptions).
- Unaligned data is significantly faster with MPP than with NPP due to my one-kernel-approach.
- Box- or Gauss-filtering with large kernels is (up to 90x) faster with MPP than with NPP. All tests were done on a RTX4090.

To date, nearly all functions of the image processing subset of NPP are implemented – except for color conversion and a few functions where I either have not yet decided on how to implement them or I simply have no idea what they are good for or how to use them.

# What is already available in MPP:

- All functionality in 
  - arithmetic
  - dataExchangeAndInit
  - geometricTransforms 
  - morphology
  - thresholdAndCompare 
  - statistics, except the circularRadialProfile functions.
  - filtering is also complete except for approximately 5 functions.
  - (colorConversion is still work in progress...)

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
- One (**C1**), two (**C2**), three (**C**3), four (**C4**) channel images and three channels plus alpha channel (**AC4**)
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
- A C++ class-based wrapper for NPP like the one in managedCuda – initially the idea was to use NPP as a reference implementation but I quickly gave that up, the code is still there though and was usefull for performance comparisons.
- All cuda kernels are also implemented in a “simpleCPU” variant using no optimization and no parallelization at all, but the same operators and functors as the cuda kernels. This simplified variant is now considered as the reference implementation, the gold standard that all computed values must reach.
- Compiles on Windows with VisualC++ and Linux with GCC 13.3+
- No dependencies other than provided in the repository (except Cuda SDK)

# Other than adding the missing functionality, there is still a lot to do:
- So far, everything is compiled to static libraries and then finally linked to one big executable... some DLLs/SOs get more and more necessary for the library to be of any practical use.
- A C-API is not there yet at all, currently MPP is only a class-based C++ API

# What is next?
- Once MPP is fully implemented with Cuda, the next logical step would be to port the cuda kernels to hip for AMD GPUs.
- Large parts of the code could also be reused for a CPU implementation using SSE or AVX, which then leads to the question: how easy is it to reach the performance of IPP on CPU? Arm’s Neon vector units would then also benefit from such an implementation…


