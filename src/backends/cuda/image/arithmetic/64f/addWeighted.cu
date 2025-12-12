#include "../addWeighted_impl.h"

using namespace mpp::cuda;

namespace mpp::image::cuda
{

ForAllChannelsWithAlphaInvokeAddWeightedSrcSrc(64f);
ForAllChannelsWithAlphaInvokeAddWeightedInplaceSrc(64f);

} // namespace mpp::image::cuda
