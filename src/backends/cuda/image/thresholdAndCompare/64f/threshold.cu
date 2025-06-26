#if MPP_ENABLE_CUDA_BACKEND

#include "../threshold_impl.h"

using namespace mpp::cuda;

namespace mpp::image::cuda
{

ForAllChannelsWithAlphaInvokeThresholdLTSrcC(64f);

ForAllChannelsWithAlphaInvokeThresholdLTSrcDevC(64f);

ForAllChannelsWithAlphaInvokeThresholdLTInplaceC(64f);

ForAllChannelsWithAlphaInvokeThresholdLTInplaceDevC(64f);

ForAllChannelsWithAlphaInvokeThresholdGTSrcC(64f);

ForAllChannelsWithAlphaInvokeThresholdGTSrcDevC(64f);

ForAllChannelsWithAlphaInvokeThresholdGTInplaceC(64f);

ForAllChannelsWithAlphaInvokeThresholdGTInplaceDevC(64f);

ForAllChannelsWithAlphaInvokeThresholdLTValSrcC(64f);

ForAllChannelsWithAlphaInvokeThresholdLTValInplaceC(64f);

ForAllChannelsWithAlphaInvokeThresholdGTValSrcC(64f);

ForAllChannelsWithAlphaInvokeThresholdGTValInplaceC(64f);

ForAllChannelsWithAlphaInvokeThresholdLTValGTValSrcC(64f);

ForAllChannelsWithAlphaInvokeThresholdLTValGTValInplaceC(64f);

} // namespace mpp::image::cuda
#endif // MPP_ENABLE_CUDA_BACKEND
