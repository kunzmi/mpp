#include "../sqr_impl.h"

using namespace mpp::cuda;

namespace mpp::image::cuda
{

ForAllChannelsNoAlphaInvokeSqrSrc(16sc);
ForAllChannelsNoAlphaInvokeSqrInplace(16sc);

} // namespace mpp::image::cuda
