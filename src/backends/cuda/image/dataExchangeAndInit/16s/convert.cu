#if OPP_ENABLE_CUDA_BACKEND

#include "../convert_impl.h"

using namespace opp::cuda;

namespace opp::image::cuda
{

ForAllChannelsWithAlphaInvokeConvert(16s, 8u);
ForAllChannelsWithAlphaInvokeConvert(16s, 16u);
ForAllChannelsWithAlphaInvokeConvert(16s, 32u);
ForAllChannelsWithAlphaInvokeConvert(16s, 32s);
ForAllChannelsWithAlphaInvokeConvert(16s, 16bf);
ForAllChannelsWithAlphaInvokeConvert(16s, 16f);
ForAllChannelsWithAlphaInvokeConvert(16s, 32f);
ForAllChannelsWithAlphaInvokeConvert(16s, 64f);
ForAllChannelsWithAlphaInvokeConvertScaleRound(16s, 8s);

} // namespace opp::image::cuda
#endif // OPP_ENABLE_CUDA_BACKEND
