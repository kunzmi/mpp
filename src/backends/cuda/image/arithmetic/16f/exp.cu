#include "../exp_impl.h"

using namespace mpp::cuda;

namespace mpp::image::cuda
{

ForAllChannelsWithAlphaInvokeExpSrc(16f);
ForAllChannelsWithAlphaInvokeExpInplace(16f);

} // namespace mpp::image::cuda
