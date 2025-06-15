#if OPP_ENABLE_CUDA_BACKEND

#include "../alphaPremul_impl.h"

using namespace opp::cuda;

namespace opp::image::cuda
{

// we treat 4th channel as alpha channel, but don't need any pre-load of alpha
InstantiateInvokeAlphaPremulSrc_For(Pixel32uC4);

// we treat 4th channel as alpha channel, but don't need any pre-load of alpha
InstantiateInvokeAlphaPremulInplace_For(Pixel32uC4);

// we treat 4th channel as alpha channel, but don't need any pre-load of alpha
InstantiateInvokeAlphaPremulACSrc_For(Pixel32uC4);

// we treat 4th channel as alpha channel, but don't need any pre-load of alpha
InstantiateInvokeAlphaPremulACInplace_For(Pixel32uC4);

} // namespace opp::image::cuda
#endif // OPP_ENABLE_CUDA_BACKEND
