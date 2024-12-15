#pragma once
#include <common/defines.h>
#include <common/image/pixelTypes.h>
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
#endif

#ifdef OPP_CUDA_TEMPLATE_REGISTRY_IS_ACTIVE

#define OPP_CUDA_REGISTER_TEMPALTE                                                                                     \
    constexpr opp::cuda::KernelNameWrapper kernelName(__PRETTY_FUNCTION__, opp::image::pixel_type_name<SrcT>::value,   \
                                                      opp::image::pixel_type_name<ComputeT>::value,                    \
                                                      opp::image::pixel_type_name<DstT>::value);                       \
    const void *_ = &opp::cuda::TemplateRegistry<kernelName>::sInstance;

inline std::vector<std::string> &GetTemplateInstances()
{
    static std::vector<std::string> templates;
    return templates;
}

// we cannot directly use char* or string as a template parameter, so use this constexpr wrapper for function name,
// SrcT, computeT and DstT
template <size_t A, size_t B, size_t C, size_t D> struct KernelNameWrapper
{
    constexpr KernelNameWrapper(const char (&aKernelName)[A], const char (&aSrcName)[B], const char (&aCompName)[C],
                                const char (&aDstName)[D])
    {
        for (size_t i = 0; i < A; i++)
        {
            kernelName[i] = aKernelName[i];
        }

        for (size_t i = 0; i < B; i++)
        {
            srcType[i] = aSrcName[i];
        }

        for (size_t i = 0; i < C; i++)
        {
            compType[i] = aCompName[i];
        }

        for (size_t i = 0; i < D; i++)
        {
            dstType[i] = aDstName[i];
        }
    }

    char kernelName[A];
    char srcType[B];
    char compType[C];
    char dstType[D];
};

template <KernelNameWrapper name> struct TemplateRegistry
{
    TemplateRegistry()
    {
        std::string finalName(name.kernelName);
        const std::string srcT("SrcT");
        const std::string computeT("ComputeT");
        const std::string dstT("DstT");

        std::string srcType(name.srcType);
        std::string computeType(name.compType);
        std::string dstType(name.dstType);

        // replace all SrcT etc by their actual type names:
        size_t pos = finalName.find(srcT);
        while (pos != std::string::npos)
        {
            finalName.replace(pos, srcT.size(), srcType);
            pos = finalName.find(srcT, pos + srcType.size());
        }

        pos = finalName.find(computeT);
        while (pos != std::string::npos)
        {
            finalName.replace(pos, computeT.size(), computeType);
            pos = finalName.find(computeT, pos + computeType.size());
        }

        pos = finalName.find(dstT);
        while (pos != std::string::npos)
        {
            finalName.replace(pos, dstT.size(), dstType);
            pos = finalName.find(dstT, pos + dstType.size());
        }

        GetTemplateInstances().push_back(finalName);
    }
    // static class member, initialized before main()
    static TemplateRegistry sInstance;
};

template <KernelNameWrapper name> TemplateRegistry<name> TemplateRegistry<name>::sInstance;
#endif
} // namespace opp::cuda