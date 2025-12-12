#include "../conj_impl.h"

using namespace mpp::cuda;

namespace mpp::image::cuda
{

ForAllChannelsNoAlphaInvokeConjSrc(32sc);
ForAllChannelsNoAlphaInvokeConjInplace(32sc);

} // namespace mpp::image::cuda
