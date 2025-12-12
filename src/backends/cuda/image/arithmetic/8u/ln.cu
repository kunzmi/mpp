#include "../ln_impl.h"

using namespace mpp::cuda;

namespace mpp::image::cuda
{

ForAllChannelsWithAlphaInvokeLnSrc(8u);
ForAllChannelsWithAlphaInvokeLnInplace(8u);

} // namespace mpp::image::cuda
