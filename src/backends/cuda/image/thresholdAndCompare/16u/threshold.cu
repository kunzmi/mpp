#if OPP_ENABLE_CUDA_BACKEND

#include "../threshold_impl.h"

using namespace opp::cuda;

namespace opp::image::cuda
{

ForAllChannelsWithAlphaInvokeThresholdLTSrcC(16u);

ForAllChannelsWithAlphaInvokeThresholdLTSrcDevC(16u);

ForAllChannelsWithAlphaInvokeThresholdLTInplaceC(16u);

ForAllChannelsWithAlphaInvokeThresholdLTInplaceDevC(16u);

ForAllChannelsWithAlphaInvokeThresholdGTSrcC(16u);

ForAllChannelsWithAlphaInvokeThresholdGTSrcDevC(16u);

ForAllChannelsWithAlphaInvokeThresholdGTInplaceC(16u);

ForAllChannelsWithAlphaInvokeThresholdGTInplaceDevC(16u);

ForAllChannelsWithAlphaInvokeThresholdLTValSrcC(16u);

ForAllChannelsWithAlphaInvokeThresholdLTValInplaceC(16u);

ForAllChannelsWithAlphaInvokeThresholdGTValSrcC(16u);

ForAllChannelsWithAlphaInvokeThresholdGTValInplaceC(16u);

ForAllChannelsWithAlphaInvokeThresholdLTValGTValSrcC(16u);

ForAllChannelsWithAlphaInvokeThresholdLTValGTValInplaceC(16u);

} // namespace opp::image::cuda
#endif // OPP_ENABLE_CUDA_BACKEND
