#include "../mirror_impl.h"

using namespace mpp::cuda;

namespace mpp::image::cuda
{

ForAllChannelsWithAlphaInvokeMirrorSrc(64f);
ForAllChannelsWithAlphaInvokeMirrorInplace(64f);

} // namespace mpp::image::cuda
