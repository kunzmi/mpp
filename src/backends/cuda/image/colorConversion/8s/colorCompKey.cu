#include "../colorCompKey_impl.h"

using namespace mpp::cuda;

namespace mpp::image::cuda
{

ForAllChannelsWithAlphaInvokeColorCompKeySrcSrc(8s);
ForAllChannelsWithAlphaInvokeColorCompKeyInplaceSrcSrc(8s);

} // namespace mpp::image::cuda
