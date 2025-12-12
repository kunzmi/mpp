#include "../compareEqEps_impl.h"

using namespace mpp::cuda;

namespace mpp::image::cuda
{

ForAllChannelsWithAlphaInvokeCompareEqEpsSrcSrc(16bf);
ForAllChannelsWithAlphaInvokeCompareEqEpsSrcC(16bf);
ForAllChannelsWithAlphaInvokeCompareEqEpsSrcDevC(16bf);

} // namespace mpp::image::cuda
