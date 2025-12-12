#include "../mirror_impl.h"

using namespace mpp::cuda;

namespace mpp::image::cuda
{

ForAllChannelsWithAlphaInvokeMirrorSrc(8u);
ForAllChannelsWithAlphaInvokeMirrorInplace(8u);

} // namespace mpp::image::cuda
