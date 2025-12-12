#include "../threshold_impl.h"

using namespace mpp::cuda;

namespace mpp::image::cuda
{

ForAllChannelsWithAlphaInvokeThresholdLTSrcC(16s);

ForAllChannelsWithAlphaInvokeThresholdLTSrcDevC(16s);

ForAllChannelsWithAlphaInvokeThresholdLTInplaceC(16s);

ForAllChannelsWithAlphaInvokeThresholdLTInplaceDevC(16s);

ForAllChannelsWithAlphaInvokeThresholdGTSrcC(16s);

ForAllChannelsWithAlphaInvokeThresholdGTSrcDevC(16s);

ForAllChannelsWithAlphaInvokeThresholdGTInplaceC(16s);

ForAllChannelsWithAlphaInvokeThresholdGTInplaceDevC(16s);

ForAllChannelsWithAlphaInvokeThresholdLTValSrcC(16s);

ForAllChannelsWithAlphaInvokeThresholdLTValInplaceC(16s);

ForAllChannelsWithAlphaInvokeThresholdGTValSrcC(16s);

ForAllChannelsWithAlphaInvokeThresholdGTValInplaceC(16s);

ForAllChannelsWithAlphaInvokeThresholdLTValGTValSrcC(16s);

ForAllChannelsWithAlphaInvokeThresholdLTValGTValInplaceC(16s);

} // namespace mpp::image::cuda
