#pragma once

#include <common/defines.h>

#if defined(IS_HOST_COMPILER) && defined(_WIN32)

#ifdef MPP_CUDAI
#define MPPEXPORT_CUDAI __declspec(dllexport)
#else
#define MPPEXPORT_CUDAI __declspec(dllimport)
#endif

#else

#define MPPEXPORT_CUDAI

#endif
