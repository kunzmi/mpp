#pragma once

#include <common/defines.h>

#if defined(IS_HOST_COMPILER) && defined(_WIN32)

#ifdef MPP_SIMPLECPU
#define MPPEXPORT_SIMPLECPU __declspec(dllexport)
#else
#define MPPEXPORT_SIMPLECPU __declspec(dllimport)
#endif

#else

#define MPPEXPORT_SIMPLECPU

#endif
