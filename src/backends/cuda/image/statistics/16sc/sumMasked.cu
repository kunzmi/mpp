#if OPP_ENABLE_CUDA_BACKEND

#include "../sumMasked_impl.h"

using namespace opp::cuda;

namespace opp::image::cuda
{

ForAllChannelsNoAlpha(16sc, 1);
ForAllChannelsNoAlpha(16sc, 2);

} // namespace opp::image::cuda
#endif // OPP_ENABLE_CUDA_BACKEND
