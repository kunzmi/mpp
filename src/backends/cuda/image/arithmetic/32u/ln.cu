#include "../ln_impl.h"

using namespace mpp::cuda;

namespace mpp::image::cuda
{

ForAllChannelsWithAlphaInvokeLnSrc(32u);
ForAllChannelsWithAlphaInvokeLnInplace(32u);

} // namespace mpp::image::cuda
