#include "../addWeightedMasked_impl.h"

using namespace mpp::cuda;

namespace mpp::image::cuda
{

ForAllChannelsWithAlphaInvokeAddWeightedSrcSrcMask(16f);
ForAllChannelsWithAlphaInvokeAddWeightedInplaceSrcMask(16f);

} // namespace mpp::image::cuda
