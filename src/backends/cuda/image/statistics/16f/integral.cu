#include "../integral_impl.h"

using namespace mpp::cuda;

namespace mpp::image::cuda
{

ForAllChannelsNoAlpha(16f, 32f);
ForAllChannelsNoAlpha(16f, 64f);

} // namespace mpp::image::cuda
