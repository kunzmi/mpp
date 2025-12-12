#include "../ln_impl.h"

using namespace mpp::cuda;

namespace mpp::image::cuda
{

ForAllChannelsWithAlphaInvokeLnSrc(32f);
ForAllChannelsWithAlphaInvokeLnInplace(32f);

} // namespace mpp::image::cuda
