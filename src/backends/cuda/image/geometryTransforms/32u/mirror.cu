#include "../mirror_impl.h"

using namespace mpp::cuda;

namespace mpp::image::cuda
{

ForAllChannelsWithAlphaInvokeMirrorSrc(32u);
ForAllChannelsWithAlphaInvokeMirrorInplace(32u);

} // namespace mpp::image::cuda
