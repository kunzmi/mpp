#include "../addWeighted_impl.h"

using namespace mpp::cuda;

namespace mpp::image::cuda
{

ForAllChannelsWithAlphaInvokeAddWeightedSrcSrc(16f);
ForAllChannelsWithAlphaInvokeAddWeightedInplaceSrc(16f);

} // namespace mpp::image::cuda
