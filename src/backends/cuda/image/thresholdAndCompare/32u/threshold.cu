#if OPP_ENABLE_CUDA_BACKEND

#include "../threshold_impl.h"

using namespace opp::cuda;

namespace opp::image::cuda
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

} // namespace opp::image::cuda
#endif // OPP_ENABLE_CUDA_BACKEND
