#if OPP_ENABLE_CUDA_BACKEND

#include "../threshold_impl.h"

using namespace opp::cuda;

namespace opp::image::cuda
{

ForAllChannelsWithAlphaInvokeThresholdLTSrcC(16bf);

ForAllChannelsWithAlphaInvokeThresholdLTSrcDevC(16bf);

ForAllChannelsWithAlphaInvokeThresholdLTInplaceC(16bf);

ForAllChannelsWithAlphaInvokeThresholdLTInplaceDevC(16bf);

ForAllChannelsWithAlphaInvokeThresholdGTSrcC(16bf);

ForAllChannelsWithAlphaInvokeThresholdGTSrcDevC(16bf);

ForAllChannelsWithAlphaInvokeThresholdGTInplaceC(16bf);

ForAllChannelsWithAlphaInvokeThresholdGTInplaceDevC(16bf);

ForAllChannelsWithAlphaInvokeThresholdLTValSrcC(16bf);

ForAllChannelsWithAlphaInvokeThresholdLTValInplaceC(16bf);

ForAllChannelsWithAlphaInvokeThresholdGTValSrcC(16bf);

ForAllChannelsWithAlphaInvokeThresholdGTValInplaceC(16bf);

ForAllChannelsWithAlphaInvokeThresholdLTValGTValSrcC(16bf);

ForAllChannelsWithAlphaInvokeThresholdLTValGTValInplaceC(16bf);

} // namespace opp::image::cuda
#endif // OPP_ENABLE_CUDA_BACKEND
