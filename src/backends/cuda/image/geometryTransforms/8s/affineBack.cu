#if OPP_ENABLE_CUDA_BACKEND

#include "../affineBack_impl.h"

using namespace opp::cuda;

namespace opp::image::cuda
{

ForAllChannelsWithAlpha(8s);
InstantiateInvokeAffineBackSrcP2ForGeomType(8s);
InstantiateInvokeAffineBackSrcP3ForGeomType(8s);
InstantiateInvokeAffineBackSrcP4ForGeomType(8s);

} // namespace opp::image::cuda
#endif // OPP_ENABLE_CUDA_BACKEND
