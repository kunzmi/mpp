#pragma once
#include <common/defines.h>
#include <common/image/pixelTypes.h>
#include <map>
#include <string>
#include <vector>

namespace opp::cuda
{
// This is a registry to keep track of all tempalte instantiations we use for OPP CUDA kernels
// All "InvokeKernel..." functions will have a static const member that will register here

// This will only be active if the #define is activated
#define OPP_CUDA_TEMPLATE_REGISTRY_IS_ACTIVE 1

#ifndef OPP_CUDA_TEMPLATE_REGISTRY_IS_ACTIVE
// empty define so that nothing is done
#define OPP_CUDA_REGISTER_TEMPALTE
#define OPP_CUDA_REGISTER_TEMPALTE_ONLY_DSTTYPE
#endif

#ifdef OPP_CUDA_TEMPLATE_REGISTRY_IS_ACTIVE

#define OPP_CUDA_REGISTER_TEMPALTE                                                                                     \
    constexpr opp::cuda::KernelNameWrapper kernelName(__func__, opp::image::pixel_type_name<SrcT>::value,              \
                                                      opp::image::pixel_type_name<ComputeT>::value,                    \
                                                      opp::image::pixel_type_name<DstT>::value);                       \
    const void *_ = &opp::cuda::TemplateRegistry<kernelName>::sInstance;

#define OPP_CUDA_REGISTER_TEMPALTE_ONLY_DSTTYPE                                                                        \
    constexpr opp::cuda::KernelNameWrapper kernelName(__func__, opp::image::pixel_type_name<DstT>::value,              \
                                                      opp::image::pixel_type_name<DstT>::value,                        \
                                                      opp::image::pixel_type_name<DstT>::value);                       \
    const void *_ = &opp::cuda::TemplateRegistry<kernelName>::sInstance;

struct oppTemplateInstance
{
    std::string srcType;
    std::string computeType;
    std::string dstType;

    oppTemplateInstance(const char *aSrcT, const char *aComputeT, const char *aDstT)
        : srcType(aSrcT), computeType(aComputeT), dstType(aDstT)
    {
    }
};

inline std::map<std::string, std::vector<oppTemplateInstance>> &GetTemplateInstances()
{
    static std::map<std::string, std::vector<oppTemplateInstance>> templates;
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
} // namespace opp::cuda