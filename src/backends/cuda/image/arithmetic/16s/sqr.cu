#include "../sqr_impl.h"

using namespace mpp::cuda;

namespace mpp::image::cuda
{

ForAllChannelsWithAlphaInvokeSqrSrc(16s);
ForAllChannelsWithAlphaInvokeSqrInplace(16s);

} // namespace mpp::image::cuda
