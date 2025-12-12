#pragma once

#include <common/defines.h>

#if defined(IS_HOST_COMPILER)

#ifdef _MSC_VER

// MSVC needs the export attribute on the first occurance, be it a forward declaration or the real definition
#ifdef MPP_NPP
#define MPPEXPORT_NPP __declspec(dllexport)
#define MPPEXPORTFWDDECL_NPP __declspec(dllexport)
#else
#define MPPEXPORT_NPP __declspec(dllimport)
#define MPPEXPORTFWDDECL_NPP __declspec(dllimport)
#endif

#else // _MSC_VER

// GCC/Clang needs the export attribute only at the definition
#ifdef MPP_NPP
#define MPPEXPORT_NPP __attribute__((visibility("default")))
#define MPPEXPORTFWDDECL_NPP
#else
#define MPPEXPORT_NPP
#define MPPEXPORTFWDDECL_NPP
#endif

#endif // _MSC_VER
#else  // defined(IS_HOST_COMPILER)
#define MPPEXPORT_NPP
#define MPPEXPORTFWDDECL_NPP
#endif
