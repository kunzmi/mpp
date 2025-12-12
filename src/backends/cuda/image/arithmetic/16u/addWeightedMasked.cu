#include "../addWeightedMasked_impl.h"

using namespace mpp::cuda;

namespace mpp::image::cuda
{

ForAllChannelsWithAlphaInvokeAddWeightedSrcSrcMask(16u);
ForAllChannelsWithAlphaInvokeAddWeightedInplaceSrcMask(16u);

} // namespace mpp::image::cuda
