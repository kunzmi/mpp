#if MPP_ENABLE_CUDA_BACKEND

#include "../scharrHoriz_impl.h"

using namespace mpp::cuda;

namespace mpp::image::cuda
{

ForAllChannelsWithAlpha(32s, 32s);

ForAllChannelsWithAlpha(16f, 16f);
ForAllChannelsWithAlpha(16bf, 16bf);
ForAllChannelsWithAlpha(32f, 32f);
ForAllChannelsWithAlpha(64f, 64f);

ForAllChannelsNoAlpha(16sc, 16sc);
ForAllChannelsNoAlpha(32sc, 32sc);
ForAllChannelsNoAlpha(32fc, 32fc);

} // namespace mpp::image::cuda
#endif // MPP_ENABLE_CUDA_BACKEND
