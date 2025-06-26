#if MPP_ENABLE_CUDA_BACKEND

#include "../threshold_impl.h"

using namespace mpp::cuda;

namespace mpp::image::cuda
{

ForAllChannelsWithAlphaInvokeThresholdLTSrcC(8s);

ForAllChannelsWithAlphaInvokeThresholdLTSrcDevC(8s);

ForAllChannelsWithAlphaInvokeThresholdLTInplaceC(8s);

ForAllChannelsWithAlphaInvokeThresholdLTInplaceDevC(8s);

ForAllChannelsWithAlphaInvokeThresholdGTSrcC(8s);

ForAllChannelsWithAlphaInvokeThresholdGTSrcDevC(8s);

ForAllChannelsWithAlphaInvokeThresholdGTInplaceC(8s);

ForAllChannelsWithAlphaInvokeThresholdGTInplaceDevC(8s);

ForAllChannelsWithAlphaInvokeThresholdLTValSrcC(8s);

ForAllChannelsWithAlphaInvokeThresholdLTValInplaceC(8s);

ForAllChannelsWithAlphaInvokeThresholdGTValSrcC(8s);

ForAllChannelsWithAlphaInvokeThresholdGTValInplaceC(8s);

ForAllChannelsWithAlphaInvokeThresholdLTValGTValSrcC(8s);

ForAllChannelsWithAlphaInvokeThresholdLTValGTValInplaceC(8s);

} // namespace mpp::image::cuda
#endif // MPP_ENABLE_CUDA_BACKEND
