#if OPP_ENABLE_CUDA_BACKEND

#include "../rshift_impl.h"

using namespace opp::cuda;

namespace opp::image::cuda
{

ForAllChannelsWithAlphaInvokeRShiftSrcC(8s);
ForAllChannelsWithAlphaInvokeRShiftInplaceC(8s);

} // namespace opp::image::cuda
#endif // OPP_ENABLE_CUDA_BACKEND
