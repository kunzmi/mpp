#include "../sqr_impl.h"

using namespace mpp::cuda;

namespace mpp::image::cuda
{

ForAllChannelsWithAlphaInvokeSqrSrc(32s);
ForAllChannelsWithAlphaInvokeSqrInplace(32s);

} // namespace mpp::image::cuda
