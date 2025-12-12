#include "../sqrt_impl.h"

using namespace mpp::cuda;

namespace mpp::image::cuda
{

ForAllChannelsWithAlphaInvokeSqrtSrc(32f);
ForAllChannelsWithAlphaInvokeSqrtInplace(32f);

} // namespace mpp::image::cuda
