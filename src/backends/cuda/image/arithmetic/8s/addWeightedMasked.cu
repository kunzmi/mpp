#include "../addWeightedMasked_impl.h"

using namespace mpp::cuda;

namespace mpp::image::cuda
{

ForAllChannelsWithAlphaInvokeAddWeightedSrcSrcMask(8s);
ForAllChannelsWithAlphaInvokeAddWeightedInplaceSrcMask(8s);

} // namespace mpp::image::cuda
