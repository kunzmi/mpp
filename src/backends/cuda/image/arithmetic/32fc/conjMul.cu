#include "../conjMul_impl.h"

using namespace mpp::cuda;

namespace mpp::image::cuda
{

ForAllChannelsNoAlphaInvokeConjMulSrcSrc(32fc);
ForAllChannelsNoAlphaInvokeConjMulInplaceSrc(32fc);

} // namespace mpp::image::cuda
