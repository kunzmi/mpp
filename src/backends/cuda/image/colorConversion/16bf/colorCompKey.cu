#include "../colorCompKey_impl.h"

using namespace mpp::cuda;

namespace mpp::image::cuda
{

ForAllChannelsWithAlphaInvokeColorCompKeySrcSrc(16bf);
ForAllChannelsWithAlphaInvokeColorCompKeyInplaceSrcSrc(16bf);

} // namespace mpp::image::cuda
