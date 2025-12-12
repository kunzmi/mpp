#include "../div_impl.h"

using namespace mpp::cuda;

namespace mpp::image::cuda
{

ForAllChannelsNoAlphaInvokeDivSrcSrcScale(16sc);
ForAllChannelsNoAlphaInvokeDivSrcCScale(16sc);
ForAllChannelsNoAlphaInvokeDivSrcDevCScale(16sc);
ForAllChannelsNoAlphaInvokeDivInplaceSrcScale(16sc);
ForAllChannelsNoAlphaInvokeDivInplaceCScale(16sc);
ForAllChannelsNoAlphaInvokeDivInplaceDevCScale(16sc);
ForAllChannelsNoAlphaInvokeDivInvInplaceSrcScale(16sc);
ForAllChannelsNoAlphaInvokeDivInvInplaceCScale(16sc);
ForAllChannelsNoAlphaInvokeDivInvInplaceDevCScale(16sc);

} // namespace mpp::image::cuda
