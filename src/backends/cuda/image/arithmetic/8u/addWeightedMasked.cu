#include "../addWeightedMasked_impl.h"

using namespace mpp::cuda;

namespace mpp::image::cuda
{

ForAllChannelsWithAlphaInvokeAddWeightedSrcSrcMask(8u);
ForAllChannelsWithAlphaInvokeAddWeightedInplaceSrcMask(8u);

} // namespace mpp::image::cuda
