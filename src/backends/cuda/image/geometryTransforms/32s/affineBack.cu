#if MPP_ENABLE_CUDA_BACKEND

#include "../affineBack_impl.h"

using namespace mpp::cuda;

namespace mpp::image::cuda
{

ForAllChannelsWithAlpha(32s);
InstantiateInvokeAffineBackSrcP2ForGeomType(32s);
InstantiateInvokeAffineBackSrcP3ForGeomType(32s);
InstantiateInvokeAffineBackSrcP4ForGeomType(32s);

} // namespace mpp::image::cuda
#endif // MPP_ENABLE_CUDA_BACKEND
