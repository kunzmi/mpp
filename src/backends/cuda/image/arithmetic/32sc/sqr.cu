#include "../sqr_impl.h"

using namespace mpp::cuda;

namespace mpp::image::cuda
{

ForAllChannelsNoAlphaInvokeSqrSrc(32sc);
ForAllChannelsNoAlphaInvokeSqrInplace(32sc);

} // namespace mpp::image::cuda
