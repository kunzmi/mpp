#if OPP_ENABLE_CUDA_BACKEND

#include "../affineBack_impl.h"

using namespace opp::cuda;

namespace opp::image::cuda
{

ForAllChannelsWithAlpha(16f);
InstantiateInvokeAffineBackSrcP2ForGeomType(16f);
InstantiateInvokeAffineBackSrcP3ForGeomType(16f);
InstantiateInvokeAffineBackSrcP4ForGeomType(16f);

} // namespace opp::image::cuda
#endif // OPP_ENABLE_CUDA_BACKEND
