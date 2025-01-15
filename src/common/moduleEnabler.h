#pragma once
#include "image/pixelTypes.h"
#include <concepts>

// define concepts and constexpr based on the C-preprocessor macros to en- or disable OPP modules
namespace opp
{
// if these macros are not set in CMAKE, fall back to enable:
#ifndef OPP_ENABLE_CUDA_BACKEND
#define OPP_ENABLE_CUDA_BACKEND 1
#endif
#ifndef OPP_ENABLE_HIP_BACKEND
#define OPP_ENABLE_HIP_BACKEND 1
#endif
#ifndef OPP_ENABLE_CPUSIMPLE_BACKEND
#define OPP_ENABLE_CPUSIMPLE_BACKEND 1
#endif
#ifndef OPP_ENABLE_NPP_BACKEND
#define OPP_ENABLE_NPP_BACKEND 1
#endif

#if OPP_ENABLE_CUDA_BACKEND
static constexpr bool enableCudaBackend = true;
#else
static constexpr bool enableCudaBackend = false;
#endif
template <typename T>
concept oppEnableCudaBackend = enableCudaBackend;

#if OPP_ENABLE_HIP_BACKEND
static constexpr bool enableHipBackend = true;
#else
static constexpr bool enableHipBackend = false;
#endif
template <typename T>
concept oppEnableHipBackend = enableHipBackend;

#if OPP_ENABLE_CPUSIMPLE_BACKEND
static constexpr bool enableCPUSimpleBackend = true;
#else
static constexpr bool enableCPUSimpleBackend = false;
#endif
template <typename T>
concept oppEnableCPUSimpleBackend = enableCPUSimpleBackend;

#if OPP_ENABLE_NPP_BACKEND
static constexpr bool enableNPPBackend = true;
#else
static constexpr bool enableNPPBackend = false;
#endif
template <typename T>
concept oppEnableNPPBackend = enableNPPBackend;

} // namespace opp