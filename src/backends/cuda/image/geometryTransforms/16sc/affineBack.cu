#if OPP_ENABLE_CUDA_BACKEND

#include "../affineBack_impl.h"

using namespace opp::cuda;

namespace opp::image::cuda
{

ForAllChannelsNoAlpha(16sc);
InstantiateInvokeAffineBackSrcP2ForGeomType(16sc);
InstantiateInvokeAffineBackSrcP3ForGeomType(16sc);
InstantiateInvokeAffineBackSrcP4ForGeomType(16sc);

} // namespace opp::image::cuda
#endif // OPP_ENABLE_CUDA_BACKEND
