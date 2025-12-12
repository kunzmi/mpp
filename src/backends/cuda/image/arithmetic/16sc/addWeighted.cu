#include "../addWeighted_impl.h"

using namespace mpp::cuda;

namespace mpp::image::cuda
{

ForAllChannelsNoAlphaInvokeAddWeightedSrcSrc(16sc);
ForAllChannelsNoAlphaInvokeAddWeightedInplaceSrc(16sc);

} // namespace mpp::image::cuda
