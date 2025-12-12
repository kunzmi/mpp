#include "../mirror_impl.h"

using namespace mpp::cuda;

namespace mpp::image::cuda
{

ForAllChannelsWithAlphaInvokeMirrorSrc(16u);
ForAllChannelsWithAlphaInvokeMirrorInplace(16u);

} // namespace mpp::image::cuda
