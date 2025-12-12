#include "../sqr_impl.h"

using namespace mpp::cuda;

namespace mpp::image::cuda
{

ForAllChannelsWithAlphaInvokeSqrSrc(32u);
ForAllChannelsWithAlphaInvokeSqrInplace(32u);

} // namespace mpp::image::cuda
