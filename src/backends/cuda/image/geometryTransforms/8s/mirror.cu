#include "../mirror_impl.h"

using namespace mpp::cuda;

namespace mpp::image::cuda
{

ForAllChannelsWithAlphaInvokeMirrorSrc(8s);
ForAllChannelsWithAlphaInvokeMirrorInplace(8s);

} // namespace mpp::image::cuda
