#include "../addWeightedMasked_impl.h"

using namespace mpp::cuda;

namespace mpp::image::cuda
{

ForAllChannelsWithAlphaInvokeAddWeightedSrcSrcMask(16s);
ForAllChannelsWithAlphaInvokeAddWeightedInplaceSrcMask(16s);

} // namespace mpp::image::cuda
