#include "../exp_impl.h"

using namespace mpp::cuda;

namespace mpp::image::cuda
{

ForAllChannelsWithAlphaInvokeExpSrc(16s);
ForAllChannelsWithAlphaInvokeExpInplace(16s);

} // namespace mpp::image::cuda
