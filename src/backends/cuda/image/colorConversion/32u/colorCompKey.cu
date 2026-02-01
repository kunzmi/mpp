#include "../colorCompKey_impl.h"

using namespace mpp::cuda;

namespace mpp::image::cuda
{

ForAllChannelsWithAlphaInvokeColorCompKeySrcSrc(32u);
ForAllChannelsWithAlphaInvokeColorCompKeyInplaceSrcSrc(32u);

} // namespace mpp::image::cuda
