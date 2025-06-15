#if OPP_ENABLE_CUDA_BACKEND

#include "../threshold_impl.h"

using namespace opp::cuda;

namespace opp::image::cuda
{

ForAllChannelsWithAlphaInvokeThresholdLTSrcC(32s);

ForAllChannelsWithAlphaInvokeThresholdLTSrcDevC(32s);

ForAllChannelsWithAlphaInvokeThresholdLTInplaceC(32s);

ForAllChannelsWithAlphaInvokeThresholdLTInplaceDevC(32s);

ForAllChannelsWithAlphaInvokeThresholdGTSrcC(32s);

ForAllChannelsWithAlphaInvokeThresholdGTSrcDevC(32s);

ForAllChannelsWithAlphaInvokeThresholdGTInplaceC(32s);

ForAllChannelsWithAlphaInvokeThresholdGTInplaceDevC(32s);

ForAllChannelsWithAlphaInvokeThresholdLTValSrcC(32s);

ForAllChannelsWithAlphaInvokeThresholdLTValInplaceC(32s);

ForAllChannelsWithAlphaInvokeThresholdGTValSrcC(32s);

ForAllChannelsWithAlphaInvokeThresholdGTValInplaceC(32s);

ForAllChannelsWithAlphaInvokeThresholdLTValGTValSrcC(32s);

ForAllChannelsWithAlphaInvokeThresholdLTValGTValInplaceC(32s);

} // namespace opp::image::cuda
#endif // OPP_ENABLE_CUDA_BACKEND
