#include "../sqr_impl.h"

using namespace mpp::cuda;

namespace mpp::image::cuda
{

ForAllChannelsNoAlphaInvokeSqrSrc(32fc);
ForAllChannelsNoAlphaInvokeSqrInplace(32fc);

} // namespace mpp::image::cuda
