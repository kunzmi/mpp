#include "../colorCompKey_impl.h"

using namespace mpp::cuda;

namespace mpp::image::cuda
{

ForAllChannelsWithAlphaInvokeColorCompKeySrcSrc(64f);
ForAllChannelsWithAlphaInvokeColorCompKeyInplaceSrcSrc(64f);

} // namespace mpp::image::cuda
