#if OPP_ENABLE_CUDA_BACKEND

#include "../affineBack_impl.h"

using namespace opp::cuda;

namespace opp::image::cuda
{

ForAllChannelsNoAlpha(32sc);
InstantiateInvokeAffineBackSrcP2ForGeomType(32sc);
InstantiateInvokeAffineBackSrcP3ForGeomType(32sc);
InstantiateInvokeAffineBackSrcP4ForGeomType(32sc);

} // namespace opp::image::cuda
#endif // OPP_ENABLE_CUDA_BACKEND
