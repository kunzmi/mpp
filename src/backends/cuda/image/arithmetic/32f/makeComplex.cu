#if OPP_ENABLE_CUDA_BACKEND

#include "../makeComplex_impl.h"

using namespace opp::cuda;

namespace opp::image::cuda
{

ForAllChannelsNoAlphaInvokeMakeComplexSrc(32f, 32fc);
ForAllChannelsNoAlphaInvokeMakeComplexSrcSrc(32f, 32fc);

} // namespace opp::image::cuda
#endif // OPP_ENABLE_CUDA_BACKEND
