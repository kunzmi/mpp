#include "../addWeighted_impl.h"

using namespace mpp::cuda;

namespace mpp::image::cuda
{

ForAllChannelsWithAlphaInvokeAddWeightedSrcSrc(32u);
ForAllChannelsWithAlphaInvokeAddWeightedInplaceSrc(32u);

} // namespace mpp::image::cuda
