#include "../conjMul_impl.h"

using namespace mpp::cuda;

namespace mpp::image::cuda
{

ForAllChannelsNoAlphaInvokeConjMulSrcSrc(16sc);
ForAllChannelsNoAlphaInvokeConjMulInplaceSrc(16sc);

} // namespace mpp::image::cuda
