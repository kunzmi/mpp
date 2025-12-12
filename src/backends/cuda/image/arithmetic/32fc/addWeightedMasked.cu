#include "../addWeightedMasked_impl.h"

using namespace mpp::cuda;

namespace mpp::image::cuda
{

ForAllChannelsNoAlphaInvokeAddWeightedSrcSrcMask(32fc);
ForAllChannelsNoAlphaInvokeAddWeightedInplaceSrcMask(32fc);

} // namespace mpp::image::cuda
