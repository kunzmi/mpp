#include "../sqr_impl.h"

using namespace mpp::cuda;

namespace mpp::image::cuda
{

ForAllChannelsWithAlphaInvokeSqrSrc(16u);
ForAllChannelsWithAlphaInvokeSqrInplace(16u);

} // namespace mpp::image::cuda
