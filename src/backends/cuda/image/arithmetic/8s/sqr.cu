#include "../sqr_impl.h"

using namespace mpp::cuda;

namespace mpp::image::cuda
{

ForAllChannelsWithAlphaInvokeSqrSrc(8s);
ForAllChannelsWithAlphaInvokeSqrInplace(8s);

} // namespace mpp::image::cuda
