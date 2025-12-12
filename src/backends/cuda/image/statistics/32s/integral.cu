#include "../integral_impl.h"

using namespace mpp::cuda;

namespace mpp::image::cuda
{

ForAllChannelsNoAlpha(32s, 32s);
ForAllChannelsNoAlpha(32s, 32f);
ForAllChannelsNoAlpha(32s, 64s);
ForAllChannelsNoAlpha(32s, 64f);

} // namespace mpp::image::cuda
