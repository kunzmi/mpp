#include "../sqr_impl.h"

using namespace mpp::cuda;

namespace mpp::image::cuda
{

ForAllChannelsWithAlphaInvokeSqrSrc(16f);
ForAllChannelsWithAlphaInvokeSqrInplace(16f);

} // namespace mpp::image::cuda
