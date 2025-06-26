#if MPP_ENABLE_CUDA_BACKEND

#include "../threshold_impl.h"

using namespace mpp::cuda;

namespace mpp::image::cuda
{

ForAllChannelsWithAlphaInvokeThresholdLTSrcC(32u);

ForAllChannelsWithAlphaInvokeThresholdLTSrcDevC(32u);

ForAllChannelsWithAlphaInvokeThresholdLTInplaceC(32u);

ForAllChannelsWithAlphaInvokeThresholdLTInplaceDevC(32u);

ForAllChannelsWithAlphaInvokeThresholdGTSrcC(32u);

ForAllChannelsWithAlphaInvokeThresholdGTSrcDevC(32u);

ForAllChannelsWithAlphaInvokeThresholdGTInplaceC(32u);

ForAllChannelsWithAlphaInvokeThresholdGTInplaceDevC(32u);

ForAllChannelsWithAlphaInvokeThresholdLTValSrcC(32u);

ForAllChannelsWithAlphaInvokeThresholdLTValInplaceC(32u);

ForAllChannelsWithAlphaInvokeThresholdGTValSrcC(32u);

ForAllChannelsWithAlphaInvokeThresholdGTValInplaceC(32u);

ForAllChannelsWithAlphaInvokeThresholdLTValGTValSrcC(32u);

ForAllChannelsWithAlphaInvokeThresholdLTValGTValInplaceC(32u);

} // namespace mpp::image::cuda
#endif // MPP_ENABLE_CUDA_BACKEND
