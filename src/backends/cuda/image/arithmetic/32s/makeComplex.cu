#include "../makeComplex_impl.h"

using namespace mpp::cuda;

namespace mpp::image::cuda
{

ForAllChannelsNoAlphaInvokeMakeComplexSrc(32s, 32sc);
ForAllChannelsNoAlphaInvokeMakeComplexSrcSrc(32s, 32sc);

} // namespace mpp::image::cuda
