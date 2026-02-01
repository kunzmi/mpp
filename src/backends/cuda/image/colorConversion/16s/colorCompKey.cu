#include "../colorCompKey_impl.h"

using namespace mpp::cuda;

namespace mpp::image::cuda
{

ForAllChannelsWithAlphaInvokeColorCompKeySrcSrc(16s);
ForAllChannelsWithAlphaInvokeColorCompKeyInplaceSrcSrc(16s);

} // namespace mpp::image::cuda
