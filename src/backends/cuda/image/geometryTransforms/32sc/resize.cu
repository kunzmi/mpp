#if MPP_ENABLE_CUDA_BACKEND

#include "../resize_impl.h"

using namespace mpp::cuda;

namespace mpp::image::cuda
{

ForAllChannelsNoAlpha(32sc);
InstantiateP2_ForGeomType(32sc);
InstantiateP3_ForGeomType(32sc);
InstantiateP4_ForGeomType(32sc);

} // namespace mpp::image::cuda
#endif // MPP_ENABLE_CUDA_BACKEND
