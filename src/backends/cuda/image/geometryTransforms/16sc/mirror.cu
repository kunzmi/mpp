#include "../mirror_impl.h"

using namespace mpp::cuda;

namespace mpp::image::cuda
{

ForAllChannelsNoAlphaInvokeMirrorSrc(16sc);
ForAllChannelsNoAlphaInvokeMirrorInplace(16sc);

} // namespace mpp::image::cuda
