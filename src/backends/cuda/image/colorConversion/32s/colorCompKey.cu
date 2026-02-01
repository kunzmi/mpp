#include "../colorCompKey_impl.h"

using namespace mpp::cuda;

namespace mpp::image::cuda
{

ForAllChannelsWithAlphaInvokeColorCompKeySrcSrc(32s);
ForAllChannelsWithAlphaInvokeColorCompKeyInplaceSrcSrc(32s);

} // namespace mpp::image::cuda
