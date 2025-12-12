#include "../exp_impl.h"

using namespace mpp::cuda;

namespace mpp::image::cuda
{

ForAllChannelsNoAlphaInvokeExpSrc(32fc);
ForAllChannelsNoAlphaInvokeExpInplace(32fc);

} // namespace mpp::image::cuda
