#include "../ln_impl.h"

using namespace mpp::cuda;

namespace mpp::image::cuda
{

ForAllChannelsNoAlphaInvokeLnSrc(32sc);
ForAllChannelsNoAlphaInvokeLnInplace(32sc);

} // namespace mpp::image::cuda
