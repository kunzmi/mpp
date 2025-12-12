#include "../exp_impl.h"

using namespace mpp::cuda;

namespace mpp::image::cuda
{

ForAllChannelsWithAlphaInvokeExpSrc(8u);
ForAllChannelsWithAlphaInvokeExpInplace(8u);

} // namespace mpp::image::cuda
