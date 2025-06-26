#pragma once
#include <common/defines.h>
#include <common/image/pixelTypes.h>
#include <map>
#include <string>
#include <vector>

namespace mpp::cuda
{
// This is a registry to keep track of all tempalte instantiations we use for MPP CUDA kernels
// All "InvokeKernel..." functions will have a static const member that will register here

// This will only be active if the #define is activated
// #define MPP_CUDA_TEMPLATE_REGISTRY_IS_ACTIVE 1

#ifndef MPP_CUDA_TEMPLATE_REGISTRY_IS_ACTIVE
// empty define so that nothing is done
#define MPP_CUDA_REGISTER_TEMPALTE
#define MPP_CUDA_REGISTER_TEMPALTE_ONLY_SRCTYPE
#define MPP_CUDA_REGISTER_TEMPALTE_SRC_DST
#define MPP_CUDA_REGISTER_TEMPALTE_ONLY_DSTTYPE
#define MPP_CUDA_REGISTER_TEMPALTE_COMPUTE_DST
#endif

#ifdef MPP_CUDA_TEMPLATE_REGISTRY_IS_ACTIVE

#define MPP_CUDA_REGISTER_TEMPALTE                                                                                     \
    constexpr mpp::cuda::KernelNameWrapper kernelName(__func__, mpp::image::pixel_type_name<SrcT>::value,              \
                                                      mpp::image::pixel_type_name<ComputeT>::value,                    \
                                                      mpp::image::pixel_type_name<DstT>::value);                       \
    const void *_ = &mpp::cuda::TemplateRegistry<kernelName>::sInstance;

#define MPP_CUDA_REGISTER_TEMPALTE_ONLY_SRCTYPE                                                                        \
    constexpr mpp::cuda::KernelNameWrapper kernelName(__func__, mpp::image::pixel_type_name<SrcT>::value, "", "");     \
    const void *_ = &mpp::cuda::TemplateRegistry<kernelName>::sInstance;

#define MPP_CUDA_REGISTER_TEMPALTE_SRC_DST                                                                             \
    constexpr mpp::cuda::KernelNameWrapper kernelName(__func__, mpp::image::pixel_type_name<SrcT>::value, "",          \
                                                      mpp::image::pixel_type_name<DstT>::value);                       \
    const void *_ = &mpp::cuda::TemplateRegistry<kernelName>::sInstance;

#define MPP_CUDA_REGISTER_TEMPALTE_ONLY_DSTTYPE                                                                        \
    constexpr mpp::cuda::KernelNameWrapper kernelName(__func__, "", "", mpp::image::pixel_type_name<DstT>::value);     \
    const void *_ = &mpp::cuda::TemplateRegistry<kernelName>::sInstance;

#define MPP_CUDA_REGISTER_TEMPALTE_COMPUTE_DST                                                                         \
    constexpr mpp::cuda::KernelNameWrapper kernelName(__func__, "", mpp::image::pixel_type_name<ComputeT>::value,      \
                                                      mpp::image::pixel_type_name<DstT>::value);                       \
    const void *_ = &mpp::cuda::TemplateRegistry<kernelName>::sInstance;

struct mppTemplateInstance
{
    std::string srcType;
    std::string computeType;
    std::string dstType;

    mppTemplateInstance(const char *aSrcT, const char *aComputeT, const char *aDstT)
        : srcType(aSrcT), computeType(aComputeT), dstType(aDstT)
    {
    }
};

inline std::map<std::string, std::vector<mppTemplateInstance>> &GetTemplateInstances()
{
    static std::map<std::string, std::vector<mppTemplateInstance>> templates;
    return templates;
}

// we cannot directly use char* or string as a template parameter, so use this constexpr wrapper for function name,
// SrcT, computeT and DstT
template <size_t sizeKernelName, size_t sizeSrcType, size_t sizeCompType, size_t sizeDstType> struct KernelNameWrapper
{
    constexpr KernelNameWrapper(const char (&aKernelName)[sizeKernelName], const char (&aSrcType)[sizeSrcType],
                                const char (&aCompType)[sizeCompType], const char (&aDstType)[sizeDstType])
    {
        std::copy(aKernelName, aKernelName + sizeKernelName, kernelName);
        std::copy(aSrcType, aSrcType + sizeSrcType, srcType);
        std::copy(aCompType, aCompType + sizeCompType, compType);
        std::copy(aDstType, aDstType + sizeDstType, dstType);
    }

    char kernelName[sizeKernelName];
    char srcType[sizeSrcType];
    char compType[sizeCompType];
    char dstType[sizeDstType];
};

template <KernelNameWrapper name> struct TemplateRegistry
{
    TemplateRegistry()
    {
        std::string finalName(name.kernelName);

        // remove the "Invoke" part of the name, if it exists, as we just want the simple name
        size_t pos = finalName.find("Invoke", 0);
        if (pos == 0)
        {
            finalName = finalName.substr(6);
        }
        GetTemplateInstances()[finalName].emplace_back(name.srcType, name.compType, name.dstType);
    }
    // static class member, initialized before main()
    static TemplateRegistry sInstance;
};

template <KernelNameWrapper name> TemplateRegistry<name> TemplateRegistry<name>::sInstance;
#endif
} // namespace mpp::cuda