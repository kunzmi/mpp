#if OPP_ENABLE_CUDA_BACKEND

#include "../addMasked_impl.h"

using namespace opp::cuda;

namespace opp::image::cuda
{

ForAllChannelsNoAlphaInvokeAddSrcSrcMask(32fc);
ForAllChannelsNoAlphaInvokeAddSrcCMask(32fc);
ForAllChannelsNoAlphaInvokeAddSrcDevCMask(32fc);
ForAllChannelsNoAlphaInvokeAddInplaceSrcMask(32fc);
ForAllChannelsNoAlphaInvokeAddInplaceCMask(32fc);
ForAllChannelsNoAlphaInvokeAddInplaceDevCMask(32fc);

} // namespace opp::image::cuda
#endif // OPP_ENABLE_CUDA_BACKEND
