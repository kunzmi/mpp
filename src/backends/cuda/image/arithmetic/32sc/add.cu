#include "../add_impl.h"

using namespace mpp::cuda;

namespace mpp::image::cuda
{

ForAllChannelsNoAlphaInvokeAddSrcSrc(32sc);
ForAllChannelsNoAlphaInvokeAddSrcSrcScale(32sc);
ForAllChannelsNoAlphaInvokeAddSrcC(32sc);
ForAllChannelsNoAlphaInvokeAddSrcCScale(32sc);
ForAllChannelsNoAlphaInvokeAddSrcDevC(32sc);
ForAllChannelsNoAlphaInvokeAddSrcDevCScale(32sc);
ForAllChannelsNoAlphaInvokeAddInplaceSrc(32sc);
ForAllChannelsNoAlphaInvokeAddInplaceSrcScale(32sc);
ForAllChannelsNoAlphaInvokeAddInplaceC(32sc);
ForAllChannelsNoAlphaInvokeAddInplaceCScale(32sc);
ForAllChannelsNoAlphaInvokeAddInplaceDevC(32sc);
ForAllChannelsNoAlphaInvokeAddInplaceDevCScale(32sc);

} // namespace mpp::image::cuda
