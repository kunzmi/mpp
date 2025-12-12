#include "../conj_impl.h"

using namespace mpp::cuda;

namespace mpp::image::cuda
{

ForAllChannelsNoAlphaInvokeConjSrc(32fc);
ForAllChannelsNoAlphaInvokeConjInplace(32fc);

} // namespace mpp::image::cuda
