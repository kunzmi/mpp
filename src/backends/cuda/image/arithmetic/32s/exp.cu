#include "../exp_impl.h"

using namespace mpp::cuda;

namespace mpp::image::cuda
{

ForAllChannelsWithAlphaInvokeExpSrc(32s);
ForAllChannelsWithAlphaInvokeExpInplace(32s);

} // namespace mpp::image::cuda
