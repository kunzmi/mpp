#include "../integral_impl.h"

using namespace mpp::cuda;

namespace mpp::image::cuda
{

ForAllChannelsNoAlpha(32u, 32s);
ForAllChannelsNoAlpha(32u, 32f);
ForAllChannelsNoAlpha(32u, 64s);
ForAllChannelsNoAlpha(32u, 64f);

} // namespace mpp::image::cuda
