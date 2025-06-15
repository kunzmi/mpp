#if OPP_ENABLE_CUDA_BACKEND

#include "../fixedSizeDilationGray_impl.h"

using namespace opp::cuda;

namespace opp::image::cuda
{

ForAllChannelsWithAlpha(16bf, 16bf);

} // namespace opp::image::cuda
#endif // OPP_ENABLE_CUDA_BACKEND
