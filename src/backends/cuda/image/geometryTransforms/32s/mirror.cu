#include "../mirror_impl.h"

using namespace mpp::cuda;

namespace mpp::image::cuda
{

ForAllChannelsWithAlphaInvokeMirrorSrc(32s);
ForAllChannelsWithAlphaInvokeMirrorInplace(32s);

} // namespace mpp::image::cuda
