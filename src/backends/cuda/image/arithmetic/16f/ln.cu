#include "../ln_impl.h"

using namespace mpp::cuda;

namespace mpp::image::cuda
{

ForAllChannelsWithAlphaInvokeLnSrc(16f);
ForAllChannelsWithAlphaInvokeLnInplace(16f);

} // namespace mpp::image::cuda
