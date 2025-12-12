#include "../abs_impl.h"

using namespace mpp::cuda;

namespace mpp::image::cuda
{

ForAllChannelsWithAlphaInvokeAbsSrc(32s);
ForAllChannelsWithAlphaInvokeAbsInplace(32s);

} // namespace mpp::image::cuda
