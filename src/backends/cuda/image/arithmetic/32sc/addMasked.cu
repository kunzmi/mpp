#include "../addMasked_impl.h"

using namespace mpp::cuda;

namespace mpp::image::cuda
{

ForAllChannelsNoAlphaInvokeAddSrcSrcMask(32sc);
ForAllChannelsNoAlphaInvokeAddSrcSrcScaleMask(32sc);
ForAllChannelsNoAlphaInvokeAddSrcCMask(32sc);
ForAllChannelsNoAlphaInvokeAddSrcCScaleMask(32sc);
ForAllChannelsNoAlphaInvokeAddSrcDevCMask(32sc);
ForAllChannelsNoAlphaInvokeAddSrcDevCScaleMask(32sc);
ForAllChannelsNoAlphaInvokeAddInplaceSrcMask(32sc);
ForAllChannelsNoAlphaInvokeAddInplaceSrcScaleMask(32sc);
ForAllChannelsNoAlphaInvokeAddInplaceCMask(32sc);
ForAllChannelsNoAlphaInvokeAddInplaceCScaleMask(32sc);
ForAllChannelsNoAlphaInvokeAddInplaceDevCMask(32sc);
ForAllChannelsNoAlphaInvokeAddInplaceDevCScaleMask(32sc);

} // namespace mpp::image::cuda
