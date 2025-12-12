#include "../affineBack_impl.h"

using namespace mpp::cuda;

namespace mpp::image::cuda
{

ForAllChannelsWithAlpha(16f);
InstantiateInvokeAffineBackSrcP2ForGeomType(16f);
InstantiateInvokeAffineBackSrcP3ForGeomType(16f);
InstantiateInvokeAffineBackSrcP4ForGeomType(16f);

} // namespace mpp::image::cuda
