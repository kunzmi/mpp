#if OPP_ENABLE_CUDA_BACKEND

#include "../conj_impl.h"

using namespace opp::cuda;

namespace opp::image::cuda
{

ForAllChannelsNoAlphaInvokeConjSrc(16sc);
ForAllChannelsNoAlphaInvokeConjInplace(16sc);

} // namespace opp::image::cuda
#endif // OPP_ENABLE_CUDA_BACKEND
