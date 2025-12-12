#include "../addWeightedMasked_impl.h"

using namespace mpp::cuda;

namespace mpp::image::cuda
{

ForAllChannelsWithAlphaInvokeAddWeightedSrcSrcMask(32u);
ForAllChannelsWithAlphaInvokeAddWeightedInplaceSrcMask(32u);

} // namespace mpp::image::cuda
