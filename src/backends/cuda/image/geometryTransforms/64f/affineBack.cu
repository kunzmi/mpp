#if OPP_ENABLE_CUDA_BACKEND

#include "../affineBack_impl.h"

using namespace opp::cuda;

namespace opp::image::cuda
{

ForAllChannelsWithAlpha(64f);
InstantiateInvokeAffineBackSrcP2ForGeomType(64f);
InstantiateInvokeAffineBackSrcP3ForGeomType(64f);
InstantiateInvokeAffineBackSrcP4ForGeomType(64f);

} // namespace opp::image::cuda
#endif // OPP_ENABLE_CUDA_BACKEND
