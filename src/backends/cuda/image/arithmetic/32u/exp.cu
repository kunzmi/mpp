#include "../exp_impl.h"

using namespace mpp::cuda;

namespace mpp::image::cuda
{

ForAllChannelsWithAlphaInvokeExpSrc(32u);
ForAllChannelsWithAlphaInvokeExpInplace(32u);

} // namespace mpp::image::cuda
