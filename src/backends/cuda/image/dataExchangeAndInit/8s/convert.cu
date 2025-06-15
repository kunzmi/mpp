#if OPP_ENABLE_CUDA_BACKEND

#include "../convert_impl.h"

using namespace opp::cuda;

namespace opp::image::cuda
{

ForAllChannelsWithAlphaInvokeConvert(8s, 8u);
ForAllChannelsWithAlphaInvokeConvert(8s, 16u);
ForAllChannelsWithAlphaInvokeConvert(8s, 16s);
ForAllChannelsWithAlphaInvokeConvert(8s, 32u);
ForAllChannelsWithAlphaInvokeConvert(8s, 32s);
ForAllChannelsWithAlphaInvokeConvert(8s, 16bf);
ForAllChannelsWithAlphaInvokeConvert(8s, 16f);
ForAllChannelsWithAlphaInvokeConvert(8s, 32f);
ForAllChannelsWithAlphaInvokeConvert(8s, 64f);

} // namespace opp::image::cuda
#endif // OPP_ENABLE_CUDA_BACKEND
