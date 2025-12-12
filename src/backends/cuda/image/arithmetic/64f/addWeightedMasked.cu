#include "../addWeightedMasked_impl.h"

using namespace mpp::cuda;

namespace mpp::image::cuda
{

ForAllChannelsWithAlphaInvokeAddWeightedSrcSrcMask(64f);
ForAllChannelsWithAlphaInvokeAddWeightedInplaceSrcMask(64f);

} // namespace mpp::image::cuda
