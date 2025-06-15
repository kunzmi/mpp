#if OPP_ENABLE_CUDA_BACKEND

#include "../rshift_impl.h"

using namespace opp::cuda;

namespace opp::image::cuda
{

ForAllChannelsWithAlphaInvokeRShiftSrcC(32s);
ForAllChannelsWithAlphaInvokeRShiftInplaceC(32s);

} // namespace opp::image::cuda
#endif // OPP_ENABLE_CUDA_BACKEND
