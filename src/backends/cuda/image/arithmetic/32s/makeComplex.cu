#if OPP_ENABLE_CUDA_BACKEND

#include "../makeComplex_impl.h"

using namespace opp::cuda;

namespace opp::image::cuda
{

ForAllChannelsNoAlphaInvokeMakeComplexSrc(32s, 32sc);
ForAllChannelsNoAlphaInvokeMakeComplexSrcSrc(32s, 32sc);

} // namespace opp::image::cuda
#endif // OPP_ENABLE_CUDA_BACKEND
