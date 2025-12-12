#include "../threshold_impl.h"

using namespace mpp::cuda;

namespace mpp::image::cuda
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

} // namespace mpp::image::cuda
