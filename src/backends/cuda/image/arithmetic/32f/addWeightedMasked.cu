#include "../addWeightedMasked_impl.h"

using namespace mpp::cuda;

namespace mpp::image::cuda
{

ForAllChannelsWithAlphaInvokeAddWeightedSrcSrcMask(32f);
ForAllChannelsWithAlphaInvokeAddWeightedInplaceSrcMask(32f);

} // namespace mpp::image::cuda
