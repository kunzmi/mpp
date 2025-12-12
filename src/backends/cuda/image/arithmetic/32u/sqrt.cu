#include "../sqrt_impl.h"

using namespace mpp::cuda;

namespace mpp::image::cuda
{

ForAllChannelsWithAlphaInvokeSqrtSrc(32u);
ForAllChannelsWithAlphaInvokeSqrtInplace(32u);

} // namespace mpp::image::cuda
