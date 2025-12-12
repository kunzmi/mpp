#include "../affineBack_impl.h"

using namespace mpp::cuda;

namespace mpp::image::cuda
{

ForAllChannelsWithAlpha(32u);
InstantiateInvokeAffineBackSrcP2ForGeomType(32u);
InstantiateInvokeAffineBackSrcP3ForGeomType(32u);
InstantiateInvokeAffineBackSrcP4ForGeomType(32u);

} // namespace mpp::image::cuda
