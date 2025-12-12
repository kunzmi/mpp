#pragma once

#include <common/defines.h>

#if defined(IS_HOST_COMPILER)

#ifdef _MSC_VER

// MSVC needs the export attribute on the first occurance, be it a forward declaration or the real definition
#ifdef MPP_CUDACORE
#define MPPEXPORT_CUDACORE __declspec(dllexport)
#define MPPEXPORTFWDDECL_CUDACORE __declspec(dllexport)
#else
#define MPPEXPORT_CUDACORE __declspec(dllimport)
#define MPPEXPORTFWDDECL_CUDACORE __declspec(dllimport)
#endif

#else // _MSC_VER

// GCC/Clang needs the export attribute only at the definition
#ifdef MPP_CUDACORE
#define MPPEXPORT_CUDACORE __attribute__((visibility("default")))
#define MPPEXPORTFWDDECL_CUDACORE
#else
#define MPPEXPORT_CUDACORE
#define MPPEXPORTFWDDECL_CUDACORE
#endif

#endif // _MSC_VER
#else  // defined(IS_HOST_COMPILER)
#define MPPEXPORT_CUDACORE
#define MPPEXPORTFWDDECL_CUDACORE
#endif
