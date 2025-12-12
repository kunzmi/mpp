#include "../makeComplex_impl.h"

using namespace mpp::cuda;

namespace mpp::image::cuda
{

ForAllChannelsNoAlphaInvokeMakeComplexSrc(32f, 32fc);
ForAllChannelsNoAlphaInvokeMakeComplexSrcSrc(32f, 32fc);

} // namespace mpp::image::cuda
