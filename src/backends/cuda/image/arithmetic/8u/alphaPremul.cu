#if MPP_ENABLE_CUDA_BACKEND

#include "../alphaPremul_impl.h"

using namespace mpp::cuda;

namespace mpp::image::cuda
{

// we treat 4th channel as alpha channel, but don't need any pre-load of alpha
InstantiateInvokeAlphaPremulSrc_For(Pixel8uC4);

// we treat 4th channel as alpha channel, but don't need any pre-load of alpha
InstantiateInvokeAlphaPremulInplace_For(Pixel8uC4);

// we treat 4th channel as alpha channel, but don't need any pre-load of alpha
InstantiateInvokeAlphaPremulACSrc_For(Pixel8uC4);

// we treat 4th channel as alpha channel, but don't need any pre-load of alpha
InstantiateInvokeAlphaPremulACInplace_For(Pixel8uC4);

} // namespace mpp::image::cuda
#endif // MPP_ENABLE_CUDA_BACKEND
