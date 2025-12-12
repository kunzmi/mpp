#include "../compareEqEps_impl.h"

using namespace mpp::cuda;

namespace mpp::image::cuda
{

ForAllChannelsWithAlphaInvokeCompareEqEpsSrcSrc(32f);
ForAllChannelsWithAlphaInvokeCompareEqEpsSrcC(32f);
ForAllChannelsWithAlphaInvokeCompareEqEpsSrcDevC(32f);

} // namespace mpp::image::cuda
