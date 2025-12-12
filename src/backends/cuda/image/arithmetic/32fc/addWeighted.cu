#include "../addWeighted_impl.h"

using namespace mpp::cuda;

namespace mpp::image::cuda
{

ForAllChannelsNoAlphaInvokeAddWeightedSrcSrc(32fc);
ForAllChannelsNoAlphaInvokeAddWeightedInplaceSrc(32fc);

} // namespace mpp::image::cuda
