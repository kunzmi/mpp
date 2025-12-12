#include "../integralSqr_impl.h"

using namespace mpp::cuda;

namespace mpp::image::cuda
{

ForAllChannelsNoAlpha(32f, 32f, 64f);
ForAllChannelsNoAlpha(32f, 64f, 64f);

} // namespace mpp::image::cuda
