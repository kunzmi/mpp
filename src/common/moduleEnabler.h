#pragma once
#include "image/pixelTypes.h"
#include <concepts>

// define concepts and constexpr based on the C-preprocessor macros to en- or disable MPP modules
namespace mpp
{
// if these macros are not set in CMAKE, fall back to enable:
#ifndef MPP_ENABLE_CUDA_BACKEND
#define MPP_ENABLE_CUDA_BACKEND 1
#endif
#ifndef MPP_ENABLE_HIP_BACKEND
#define MPP_ENABLE_HIP_BACKEND 1
#endif
#ifndef MPP_ENABLE_CPUSIMPLE_BACKEND
#define MPP_ENABLE_CPUSIMPLE_BACKEND 1
#endif
#ifndef MPP_ENABLE_NPP_BACKEND
#define MPP_ENABLE_NPP_BACKEND 1
#endif

#if MPP_ENABLE_CUDA_BACKEND
static constexpr bool enableCudaBackend = true;
#else
static constexpr bool enableCudaBackend = false;
#endif
template <typename T>
concept mppEnableCudaBackend = enableCudaBackend;

#if MPP_ENABLE_HIP_BACKEND
static constexpr bool enableHipBackend = true;
#else
static constexpr bool enableHipBackend = false;
#endif
template <typename T>
concept mppEnableHipBackend = enableHipBackend;

#if MPP_ENABLE_CPUSIMPLE_BACKEND
static constexpr bool enableCPUSimpleBackend = true;
#else
static constexpr bool enableCPUSimpleBackend = false;
#endif
template <typename T>
concept mppEnableCPUSimpleBackend = enableCPUSimpleBackend;

#if MPP_ENABLE_NPP_BACKEND
static constexpr bool enableNPPBackend = true;
#else
static constexpr bool enableNPPBackend = false;
#endif
template <typename T>
concept mppEnableNPPBackend = enableNPPBackend;

} // namespace mpp