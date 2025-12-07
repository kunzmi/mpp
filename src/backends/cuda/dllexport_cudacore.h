#pragma once

#include <common/defines.h>

#if defined(IS_HOST_COMPILER) && defined(_WIN32)

#ifdef MPP_CUDACORE
#define MPPEXPORT_CUDACORE __declspec(dllexport)
#else
#define MPPEXPORT_CUDACORE __declspec(dllimport)
#endif

#else

#define MPPEXPORT_CUDACORE

#endif
