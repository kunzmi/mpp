#include "../ln_impl.h"

using namespace mpp::cuda;

namespace mpp::image::cuda
{

ForAllChannelsWithAlphaInvokeLnSrc(16u);
ForAllChannelsWithAlphaInvokeLnInplace(16u);

} // namespace mpp::image::cuda
