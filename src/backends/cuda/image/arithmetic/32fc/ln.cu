#include "../ln_impl.h"

using namespace mpp::cuda;

namespace mpp::image::cuda
{

ForAllChannelsNoAlphaInvokeLnSrc(32fc);
ForAllChannelsNoAlphaInvokeLnInplace(32fc);

} // namespace mpp::image::cuda
