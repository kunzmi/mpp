#include "../threshold_impl.h"

using namespace mpp::cuda;

namespace mpp::image::cuda
{

ForAllChannelsWithAlphaInvokeThresholdLTSrcC(32f);

ForAllChannelsWithAlphaInvokeThresholdLTSrcDevC(32f);

ForAllChannelsWithAlphaInvokeThresholdLTInplaceC(32f);

ForAllChannelsWithAlphaInvokeThresholdLTInplaceDevC(32f);

ForAllChannelsWithAlphaInvokeThresholdGTSrcC(32f);

ForAllChannelsWithAlphaInvokeThresholdGTSrcDevC(32f);

ForAllChannelsWithAlphaInvokeThresholdGTInplaceC(32f);

ForAllChannelsWithAlphaInvokeThresholdGTInplaceDevC(32f);

ForAllChannelsWithAlphaInvokeThresholdLTValSrcC(32f);

ForAllChannelsWithAlphaInvokeThresholdLTValInplaceC(32f);

ForAllChannelsWithAlphaInvokeThresholdGTValSrcC(32f);

ForAllChannelsWithAlphaInvokeThresholdGTValInplaceC(32f);

ForAllChannelsWithAlphaInvokeThresholdLTValGTValSrcC(32f);

ForAllChannelsWithAlphaInvokeThresholdLTValGTValInplaceC(32f);

} // namespace mpp::image::cuda
