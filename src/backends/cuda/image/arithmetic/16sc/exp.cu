#include "../exp_impl.h"

using namespace mpp::cuda;

namespace mpp::image::cuda
{

ForAllChannelsNoAlphaInvokeExpSrc(16sc);
ForAllChannelsNoAlphaInvokeExpInplace(16sc);

} // namespace mpp::image::cuda
