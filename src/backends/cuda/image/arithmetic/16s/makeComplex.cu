#if OPP_ENABLE_CUDA_BACKEND

#include "../makeComplex_impl.h"

using namespace opp::cuda;

namespace opp::image::cuda
{

ForAllChannelsNoAlphaInvokeMakeComplexSrc(16s, 16sc);
ForAllChannelsNoAlphaInvokeMakeComplexSrcSrc(16s, 16sc);

} // namespace opp::image::cuda
#endif // OPP_ENABLE_CUDA_BACKEND
