#if OPP_ENABLE_CUDA_BACKEND

#include "../lshift_impl.h"

using namespace opp::cuda;

namespace opp::image::cuda
{

ForAllChannelsWithAlphaInvokeLShiftSrcC(16u);
ForAllChannelsWithAlphaInvokeLShiftInplaceC(16u);

} // namespace opp::image::cuda
#endif // OPP_ENABLE_CUDA_BACKEND
