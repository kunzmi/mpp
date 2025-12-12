#include "../sqr_impl.h"

using namespace mpp::cuda;

namespace mpp::image::cuda
{

ForAllChannelsWithAlphaInvokeSqrSrc(64f);
ForAllChannelsWithAlphaInvokeSqrInplace(64f);

} // namespace mpp::image::cuda
