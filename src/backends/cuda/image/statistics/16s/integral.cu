#include "../integral_impl.h"

using namespace mpp::cuda;

namespace mpp::image::cuda
{

ForAllChannelsNoAlpha(16s, 32s);
ForAllChannelsNoAlpha(16s, 32f);
ForAllChannelsNoAlpha(16s, 64s);
ForAllChannelsNoAlpha(16s, 64f);

} // namespace mpp::image::cuda
