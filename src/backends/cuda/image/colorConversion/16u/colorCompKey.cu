#include "../colorCompKey_impl.h"

using namespace mpp::cuda;

namespace mpp::image::cuda
{

ForAllChannelsWithAlphaInvokeColorCompKeySrcSrc(16u);
ForAllChannelsWithAlphaInvokeColorCompKeyInplaceSrcSrc(16u);

} // namespace mpp::image::cuda
