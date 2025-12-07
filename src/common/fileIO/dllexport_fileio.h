#pragma once

#include <common/defines.h>

#if defined(IS_HOST_COMPILER) && defined(_WIN32)

#ifdef MPP_COMMON_FILEIO
#define MPPEXPORT_COMMON_FILEIO __declspec(dllexport)
#else
#define MPPEXPORT_COMMON_FILEIO __declspec(dllimport)
#endif

#else

#define MPPEXPORT_COMMON_FILEIO

#endif
