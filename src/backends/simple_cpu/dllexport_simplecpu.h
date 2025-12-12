#pragma once

#include <common/defines.h>

#if defined(IS_HOST_COMPILER)

#ifdef _MSC_VER

#ifdef MPP_SIMPLECPU
#define MPPEXPORT_SIMPLECPU __declspec(dllexport)
#else
#define MPPEXPORT_SIMPLECPU __declspec(dllimport)
#endif

#else // _MSC_VER

#ifdef MPP_SIMPLECPU
#define MPPEXPORT_SIMPLECPU __attribute__((visibility("default")))
#else
#define MPPEXPORT_SIMPLECPU
#endif

#endif // _MSC_VER
#else  // defined(IS_HOST_COMPILER)
#define MPPEXPORT_SIMPLECPU
#endif
