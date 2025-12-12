#include "../sqrt_impl.h"

using namespace mpp::cuda;

namespace mpp::image::cuda
{

ForAllChannelsWithAlphaInvokeSqrtSrc(8s);
ForAllChannelsWithAlphaInvokeSqrtInplace(8s);

} // namespace mpp::image::cuda
