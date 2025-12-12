#include "../exp_impl.h"

using namespace mpp::cuda;

namespace mpp::image::cuda
{

ForAllChannelsWithAlphaInvokeExpSrc(64f);
ForAllChannelsWithAlphaInvokeExpInplace(64f);

} // namespace mpp::image::cuda
