#if MPP_ENABLE_CUDA_BACKEND

#include "../affineBack_impl.h"

using namespace mpp::cuda;

namespace mpp::image::cuda
{

ForAllChannelsWithAlpha(16u);
InstantiateInvokeAffineBackSrcP2ForGeomType(16u);
InstantiateInvokeAffineBackSrcP3ForGeomType(16u);
InstantiateInvokeAffineBackSrcP4ForGeomType(16u);

} // namespace mpp::image::cuda
#endif // MPP_ENABLE_CUDA_BACKEND
