#include "../test_impl.h"

using namespace mpp::cuda;

namespace mpp::image::cuda
{

ForAllChannelsWithAlphaInvokeTestSrc(8u);
ForAllChannelsWithAlphaInvokeTestInplace(8u);

} // namespace mpp::image::cuda
