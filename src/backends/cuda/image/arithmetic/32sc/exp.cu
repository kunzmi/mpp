#include "../exp_impl.h"

using namespace mpp::cuda;

namespace mpp::image::cuda
{

ForAllChannelsNoAlphaInvokeExpSrc(32sc);
ForAllChannelsNoAlphaInvokeExpInplace(32sc);

} // namespace mpp::image::cuda
