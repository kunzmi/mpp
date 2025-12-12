#include "../integralSqr_impl.h"

using namespace mpp::cuda;

namespace mpp::image::cuda
{

ForAllChannelsNoAlpha(8u, 32s, 32s);
ForAllChannelsNoAlpha(8u, 32f, 64f);
ForAllChannelsNoAlpha(8u, 32s, 64s);
ForAllChannelsNoAlpha(8u, 64f, 64f);

} // namespace mpp::image::cuda
