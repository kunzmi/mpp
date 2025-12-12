#include "../mirror_impl.h"

using namespace mpp::cuda;

namespace mpp::image::cuda
{

ForAllChannelsWithAlphaInvokeMirrorSrc(16s);
ForAllChannelsWithAlphaInvokeMirrorInplace(16s);

} // namespace mpp::image::cuda
