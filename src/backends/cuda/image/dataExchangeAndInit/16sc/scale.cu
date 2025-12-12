#include "../scale_impl.h"

using namespace mpp::cuda;

namespace mpp::image::cuda
{

ForAllChannelsNoAlphaIntDiv(16sc, 32sc);
ForAllChannelsNoAlphaFloat(16sc, 32fc);

} // namespace mpp::image::cuda
