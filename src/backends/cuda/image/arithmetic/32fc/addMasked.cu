#include "../addMasked_impl.h"

using namespace mpp::cuda;

namespace mpp::image::cuda
{

ForAllChannelsNoAlphaInvokeAddSrcSrcMask(32fc);
ForAllChannelsNoAlphaInvokeAddSrcCMask(32fc);
ForAllChannelsNoAlphaInvokeAddSrcDevCMask(32fc);
ForAllChannelsNoAlphaInvokeAddInplaceSrcMask(32fc);
ForAllChannelsNoAlphaInvokeAddInplaceCMask(32fc);
ForAllChannelsNoAlphaInvokeAddInplaceDevCMask(32fc);

} // namespace mpp::image::cuda
