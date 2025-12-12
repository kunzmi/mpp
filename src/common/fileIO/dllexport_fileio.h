#pragma once

#include <common/defines.h>

#if defined(IS_HOST_COMPILER)

#ifdef _MSC_VER

#ifdef MPP_COMMON_FILEIO
#define MPPEXPORT_COMMON_FILEIO __declspec(dllexport)
#else
#define MPPEXPORT_COMMON_FILEIO __declspec(dllimport)
#endif

#else // _MSC_VER

#ifdef MPP_COMMON_FILEIO
#define MPPEXPORT_COMMON_FILEIO __attribute__((visibility("default")))
#else
#define MPPEXPORT_COMMON_FILEIO
#endif

#endif // _MSC_VER
#else  // defined(IS_HOST_COMPILER)
#define MPPEXPORT_COMMON_FILEIO
#endif
