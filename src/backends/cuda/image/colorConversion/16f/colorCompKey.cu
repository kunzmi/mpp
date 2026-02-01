#include "../colorCompKey_impl.h"

using namespace mpp::cuda;

namespace mpp::image::cuda
{

ForAllChannelsWithAlphaInvokeColorCompKeySrcSrc(16f);
ForAllChannelsWithAlphaInvokeColorCompKeyInplaceSrcSrc(16f);

} // namespace mpp::image::cuda
