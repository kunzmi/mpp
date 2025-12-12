#include "../ln_impl.h"

using namespace mpp::cuda;

namespace mpp::image::cuda
{

ForAllChannelsWithAlphaInvokeLnSrc(16bf);
ForAllChannelsWithAlphaInvokeLnInplace(16bf);

} // namespace mpp::image::cuda
