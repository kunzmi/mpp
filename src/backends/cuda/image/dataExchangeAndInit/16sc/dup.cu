#include "../dup_impl.h"

using namespace mpp::cuda;

namespace mpp::image::cuda
{

ForAllChannelsNoAlpha(16sc);
ForAllChannelsNoAlpha(32sc);
ForAllChannelsNoAlpha(32fc);

} // namespace mpp::image::cuda
