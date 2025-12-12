#include "../ln_impl.h"

using namespace mpp::cuda;

namespace mpp::image::cuda
{

ForAllChannelsWithAlphaInvokeLnSrc(8s);
ForAllChannelsWithAlphaInvokeLnInplace(8s);

} // namespace mpp::image::cuda
