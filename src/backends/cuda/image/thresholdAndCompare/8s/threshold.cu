#if OPP_ENABLE_CUDA_BACKEND

#include "../threshold_impl.h"

using namespace opp::cuda;

namespace opp::image::cuda
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

} // namespace opp::image::cuda
#endif // OPP_ENABLE_CUDA_BACKEND
