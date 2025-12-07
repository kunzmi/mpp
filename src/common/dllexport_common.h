#pragma once

#include <common/defines.h>

#if defined(IS_HOST_COMPILER) && defined(_WIN32)

#ifdef MPP_COMMON
#define MPPEXPORT_COMMON __declspec(dllexport)
#else
#define MPPEXPORT_COMMON __declspec(dllimport)
#endif

#else

#define MPPEXPORT_COMMON

#endif
