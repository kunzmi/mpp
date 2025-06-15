#if OPP_ENABLE_CUDA_BACKEND

#include "../lshift_impl.h"

using namespace opp::cuda;

namespace opp::image::cuda
{

ForAllChannelsWithAlphaInvokeLShiftSrcC(32s);
ForAllChannelsWithAlphaInvokeLShiftInplaceC(32s);

} // namespace opp::image::cuda
#endif // OPP_ENABLE_CUDA_BACKEND
