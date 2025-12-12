#include "../abs_impl.h"

using namespace mpp::cuda;

namespace mpp::image::cuda
{

ForAllChannelsWithAlphaInvokeAbsSrc(8s);
ForAllChannelsWithAlphaInvokeAbsInplace(8s);

} // namespace mpp::image::cuda
