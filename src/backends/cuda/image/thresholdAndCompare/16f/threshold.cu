#if OPP_ENABLE_CUDA_BACKEND

#include "../threshold_impl.h"

using namespace opp::cuda;

namespace opp::image::cuda
{

ForAllChannelsWithAlphaInvokeThresholdLTSrcC(16f);

ForAllChannelsWithAlphaInvokeThresholdLTSrcDevC(16f);

ForAllChannelsWithAlphaInvokeThresholdLTInplaceC(16f);

ForAllChannelsWithAlphaInvokeThresholdLTInplaceDevC(16f);

ForAllChannelsWithAlphaInvokeThresholdGTSrcC(16f);

ForAllChannelsWithAlphaInvokeThresholdGTSrcDevC(16f);

ForAllChannelsWithAlphaInvokeThresholdGTInplaceC(16f);

ForAllChannelsWithAlphaInvokeThresholdGTInplaceDevC(16f);

ForAllChannelsWithAlphaInvokeThresholdLTValSrcC(16f);

ForAllChannelsWithAlphaInvokeThresholdLTValInplaceC(16f);

ForAllChannelsWithAlphaInvokeThresholdGTValSrcC(16f);

ForAllChannelsWithAlphaInvokeThresholdGTValInplaceC(16f);

ForAllChannelsWithAlphaInvokeThresholdLTValGTValSrcC(16f);

ForAllChannelsWithAlphaInvokeThresholdLTValGTValInplaceC(16f);

} // namespace opp::image::cuda
#endif // OPP_ENABLE_CUDA_BACKEND
