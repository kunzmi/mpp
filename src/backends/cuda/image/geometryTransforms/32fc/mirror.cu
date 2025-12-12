#include "../mirror_impl.h"

using namespace mpp::cuda;

namespace mpp::image::cuda
{

ForAllChannelsNoAlphaInvokeMirrorSrc(32fc);
ForAllChannelsNoAlphaInvokeMirrorInplace(32fc);

} // namespace mpp::image::cuda
