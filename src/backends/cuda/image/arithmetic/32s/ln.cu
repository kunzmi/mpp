#include "../ln_impl.h"

using namespace mpp::cuda;

namespace mpp::image::cuda
{

ForAllChannelsWithAlphaInvokeLnSrc(32s);
ForAllChannelsWithAlphaInvokeLnInplace(32s);

} // namespace mpp::image::cuda
