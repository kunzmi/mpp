#include "../colorCompKey_impl.h"

using namespace mpp::cuda;

namespace mpp::image::cuda
{

ForAllChannelsWithAlphaInvokeColorCompKeySrcSrc(8u);
ForAllChannelsWithAlphaInvokeColorCompKeyInplaceSrcSrc(8u);

} // namespace mpp::image::cuda
