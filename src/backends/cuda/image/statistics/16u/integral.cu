#include "../integral_impl.h"

using namespace mpp::cuda;

namespace mpp::image::cuda
{

ForAllChannelsNoAlpha(16u, 32s);
ForAllChannelsNoAlpha(16u, 32f);
ForAllChannelsNoAlpha(16u, 64s);
ForAllChannelsNoAlpha(16u, 64f);

} // namespace mpp::image::cuda
