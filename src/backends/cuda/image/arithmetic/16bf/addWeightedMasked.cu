#include "../addWeightedMasked_impl.h"

using namespace mpp::cuda;

namespace mpp::image::cuda
{

ForAllChannelsWithAlphaInvokeAddWeightedSrcSrcMask(16bf);
ForAllChannelsWithAlphaInvokeAddWeightedInplaceSrcMask(16bf);

} // namespace mpp::image::cuda
