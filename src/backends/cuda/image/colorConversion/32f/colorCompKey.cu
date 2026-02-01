#include "../colorCompKey_impl.h"

using namespace mpp::cuda;

namespace mpp::image::cuda
{

ForAllChannelsWithAlphaInvokeColorCompKeySrcSrc(32f);
ForAllChannelsWithAlphaInvokeColorCompKeyInplaceSrcSrc(32f);

} // namespace mpp::image::cuda
