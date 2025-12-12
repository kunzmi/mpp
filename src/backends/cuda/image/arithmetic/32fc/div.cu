#include "../div_impl.h"

using namespace mpp::cuda;

namespace mpp::image::cuda
{

ForAllChannelsNoAlphaInvokeDivSrcSrc(32fc);
ForAllChannelsNoAlphaInvokeDivSrcC(32fc);
ForAllChannelsNoAlphaInvokeDivSrcDevC(32fc);
ForAllChannelsNoAlphaInvokeDivInplaceSrc(32fc);
ForAllChannelsNoAlphaInvokeDivInplaceC(32fc);
ForAllChannelsNoAlphaInvokeDivInplaceDevC(32fc);
ForAllChannelsNoAlphaInvokeDivInvInplaceSrc(32fc);
ForAllChannelsNoAlphaInvokeDivInvInplaceC(32fc);
ForAllChannelsNoAlphaInvokeDivInvInplaceDevC(32fc);

} // namespace mpp::image::cuda
