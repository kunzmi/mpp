#if OPP_ENABLE_CUDA_BACKEND

#include "../convert_impl.h"

using namespace opp::cuda;

namespace opp::image::cuda
{

ForAllChannelsWithAlphaInvokeConvert(8u, 16u);
ForAllChannelsWithAlphaInvokeConvert(8u, 16s);
ForAllChannelsWithAlphaInvokeConvert(8u, 32s);
ForAllChannelsWithAlphaInvokeConvert(8u, 16bf);
ForAllChannelsWithAlphaInvokeConvert(8u, 16f);
ForAllChannelsWithAlphaInvokeConvert(8u, 32f);
ForAllChannelsWithAlphaInvokeConvert(8u, 64f);
ForAllChannelsWithAlphaInvokeConvertScaleRound(8u, 8s);

} // namespace opp::image::cuda
#endif // OPP_ENABLE_CUDA_BACKEND
