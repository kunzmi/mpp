#pragma once

#include <common/defines.h>

#if defined(IS_HOST_COMPILER) && defined(_WIN32)

#ifdef MPP_NPP
#define MPPEXPORT_NPP __declspec(dllexport)
#else
#define MPPEXPORT_NPP __declspec(dllimport)
#endif

#else

#define MPPEXPORT_NPP

#endif
