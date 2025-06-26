#if MPP_ENABLE_CUDA_BACKEND

#include "../threshold_impl.h"

using namespace mpp::cuda;

namespace mpp::image::cuda
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

} // namespace mpp::image::cuda
#endif // MPP_ENABLE_CUDA_BACKEND
