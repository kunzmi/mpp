#include "../exp_impl.h"

using namespace mpp::cuda;

namespace mpp::image::cuda
{

ForAllChannelsWithAlphaInvokeExpSrc(32f);
ForAllChannelsWithAlphaInvokeExpInplace(32f);

} // namespace mpp::image::cuda
