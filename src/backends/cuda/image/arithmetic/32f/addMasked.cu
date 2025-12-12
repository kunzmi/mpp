#include "../addMasked_impl.h"

using namespace mpp::cuda;

namespace mpp::image::cuda
{

ForAllChannelsWithAlphaInvokeAddSrcSrcMask(32f);
ForAllChannelsWithAlphaInvokeAddSrcCMask(32f);
ForAllChannelsWithAlphaInvokeAddSrcDevCMask(32f);
ForAllChannelsWithAlphaInvokeAddInplaceSrcMask(32f);
ForAllChannelsWithAlphaInvokeAddInplaceCMask(32f);
ForAllChannelsWithAlphaInvokeAddInplaceDevCMask(32f);

} // namespace mpp::image::cuda
