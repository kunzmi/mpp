#if OPP_ENABLE_CUDA_BACKEND

#include "../threshold_impl.h"

using namespace opp::cuda;

namespace opp::image::cuda
{

ForAllChannelsWithAlphaInvokeThresholdLTSrcC(8u);

ForAllChannelsWithAlphaInvokeThresholdLTSrcDevC(8u);

ForAllChannelsWithAlphaInvokeThresholdLTInplaceC(8u);

ForAllChannelsWithAlphaInvokeThresholdLTInplaceDevC(8u);

ForAllChannelsWithAlphaInvokeThresholdGTSrcC(8u);

ForAllChannelsWithAlphaInvokeThresholdGTSrcDevC(8u);

ForAllChannelsWithAlphaInvokeThresholdGTInplaceC(8u);

ForAllChannelsWithAlphaInvokeThresholdGTInplaceDevC(8u);

ForAllChannelsWithAlphaInvokeThresholdLTValSrcC(8u);

ForAllChannelsWithAlphaInvokeThresholdLTValInplaceC(8u);

ForAllChannelsWithAlphaInvokeThresholdGTValSrcC(8u);

ForAllChannelsWithAlphaInvokeThresholdGTValInplaceC(8u);

ForAllChannelsWithAlphaInvokeThresholdLTValGTValSrcC(8u);

ForAllChannelsWithAlphaInvokeThresholdLTValGTValInplaceC(8u);

} // namespace opp::image::cuda
#endif // OPP_ENABLE_CUDA_BACKEND
