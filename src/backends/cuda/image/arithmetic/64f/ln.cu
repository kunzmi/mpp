#include "../ln_impl.h"

using namespace mpp::cuda;

namespace mpp::image::cuda
{

ForAllChannelsWithAlphaInvokeLnSrc(64f);
ForAllChannelsWithAlphaInvokeLnInplace(64f);

} // namespace mpp::image::cuda
