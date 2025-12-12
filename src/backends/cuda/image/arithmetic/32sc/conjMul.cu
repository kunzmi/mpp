#include "../conjMul_impl.h"

using namespace mpp::cuda;

namespace mpp::image::cuda
{

ForAllChannelsNoAlphaInvokeConjMulSrcSrc(32sc);
ForAllChannelsNoAlphaInvokeConjMulInplaceSrc(32sc);

} // namespace mpp::image::cuda
