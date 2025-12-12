#include "../affineBack_impl.h"

using namespace mpp::cuda;

namespace mpp::image::cuda
{

ForAllChannelsWithAlpha(32f);
InstantiateInvokeAffineBackSrcP2ForGeomType(32f);
InstantiateInvokeAffineBackSrcP3ForGeomType(32f);
InstantiateInvokeAffineBackSrcP4ForGeomType(32f);

} // namespace mpp::image::cuda
