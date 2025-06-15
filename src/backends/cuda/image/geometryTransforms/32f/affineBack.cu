#if OPP_ENABLE_CUDA_BACKEND

#include "../affineBack_impl.h"

using namespace opp::cuda;

namespace opp::image::cuda
{

ForAllChannelsWithAlpha(32f);
InstantiateInvokeAffineBackSrcP2ForGeomType(32f);
InstantiateInvokeAffineBackSrcP3ForGeomType(32f);
InstantiateInvokeAffineBackSrcP4ForGeomType(32f);

} // namespace opp::image::cuda
#endif // OPP_ENABLE_CUDA_BACKEND
