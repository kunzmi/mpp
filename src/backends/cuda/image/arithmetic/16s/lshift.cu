#if OPP_ENABLE_CUDA_BACKEND

#include "../lshift_impl.h"

using namespace opp::cuda;

namespace opp::image::cuda
{

ForAllChannelsWithAlphaInvokeLShiftSrcC(16s);
ForAllChannelsWithAlphaInvokeLShiftInplaceC(16s);

} // namespace opp::image::cuda
#endif // OPP_ENABLE_CUDA_BACKEND
