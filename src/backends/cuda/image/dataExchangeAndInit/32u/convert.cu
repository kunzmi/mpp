#if OPP_ENABLE_CUDA_BACKEND

#include "../convert_impl.h"

using namespace opp::cuda;

namespace opp::image::cuda
{

ForAllChannelsWithAlphaInvokeConvert(32u, 16bf);
ForAllChannelsWithAlphaInvokeConvert(32u, 16f);
ForAllChannelsWithAlphaInvokeConvert(32u, 32f);
ForAllChannelsWithAlphaInvokeConvert(32u, 64f);
ForAllChannelsWithAlphaInvokeConvertScaleRound(32u, 8u);
ForAllChannelsWithAlphaInvokeConvertScaleRound(32u, 8s);
ForAllChannelsWithAlphaInvokeConvertScaleRound(32u, 16u);
ForAllChannelsWithAlphaInvokeConvertScaleRound(32u, 16s);
ForAllChannelsWithAlphaInvokeConvertScaleRound(32u, 32s);

} // namespace opp::image::cuda
#endif // OPP_ENABLE_CUDA_BACKEND
