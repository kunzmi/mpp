#if OPP_ENABLE_CUDA_BACKEND

#include "../imag_impl.h"

using namespace opp::cuda;

namespace opp::image::cuda
{

ForAllChannelsNoAlpha(32fc, 32f);

} // namespace opp::image::cuda
#endif // OPP_ENABLE_CUDA_BACKEND
