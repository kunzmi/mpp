#include "../addWeightedMasked_impl.h"

using namespace mpp::cuda;

namespace mpp::image::cuda
{

ForAllChannelsWithAlphaInvokeAddWeightedSrcSrcMask(32s);
ForAllChannelsWithAlphaInvokeAddWeightedInplaceSrcMask(32s);

} // namespace mpp::image::cuda
