#include "../mirror_impl.h"

using namespace mpp::cuda;

namespace mpp::image::cuda
{

ForAllChannelsNoAlphaInvokeMirrorSrc(32sc);
ForAllChannelsNoAlphaInvokeMirrorInplace(32sc);

} // namespace mpp::image::cuda
