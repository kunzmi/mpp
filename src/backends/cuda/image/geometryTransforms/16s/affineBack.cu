#if MPP_ENABLE_CUDA_BACKEND

#include "../affineBack_impl.h"

using namespace mpp::cuda;

namespace mpp::image::cuda
{

ForAllChannelsWithAlpha(16s);
InstantiateInvokeAffineBackSrcP2ForGeomType(16s);
InstantiateInvokeAffineBackSrcP3ForGeomType(16s);
InstantiateInvokeAffineBackSrcP4ForGeomType(16s);

} // namespace mpp::image::cuda
#endif // MPP_ENABLE_CUDA_BACKEND
