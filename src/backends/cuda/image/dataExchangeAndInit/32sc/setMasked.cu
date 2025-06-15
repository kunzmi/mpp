#if OPP_ENABLE_CUDA_BACKEND

#include "../setMasked_impl.h"

using namespace opp::cuda;

namespace opp::image::cuda
{

ForAllChannelsNoAlphaInvokeSetCMask(32sc);
ForAllChannelsNoAlphaInvokeSetDevCMask(32sc);

} // namespace opp::image::cuda
#endif // OPP_ENABLE_CUDA_BACKEND
