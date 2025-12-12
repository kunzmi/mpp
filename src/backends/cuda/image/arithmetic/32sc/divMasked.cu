#include "../divMasked_impl.h"

using namespace mpp::cuda;

namespace mpp::image::cuda
{

ForAllChannelsNoAlphaInvokeDivSrcSrcScaleMask(32sc);
ForAllChannelsNoAlphaInvokeDivSrcCScaleMask(32sc);
ForAllChannelsNoAlphaInvokeDivSrcDevCScaleMask(32sc);
ForAllChannelsNoAlphaInvokeDivInplaceSrcScaleMask(32sc);
ForAllChannelsNoAlphaInvokeDivInplaceCScaleMask(32sc);
ForAllChannelsNoAlphaInvokeDivInplaceDevCScaleMask(32sc);
ForAllChannelsNoAlphaInvokeDivInvInplaceSrcScaleMask(32sc);
ForAllChannelsNoAlphaInvokeDivInvInplaceCScaleMask(32sc);
ForAllChannelsNoAlphaInvokeDivInvInplaceDevCScaleMask(32sc);

} // namespace mpp::image::cuda
