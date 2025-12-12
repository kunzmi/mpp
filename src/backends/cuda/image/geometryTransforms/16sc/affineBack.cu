#include "../affineBack_impl.h"

using namespace mpp::cuda;

namespace mpp::image::cuda
{

ForAllChannelsNoAlpha(16sc);
InstantiateInvokeAffineBackSrcP2ForGeomType(16sc);
InstantiateInvokeAffineBackSrcP3ForGeomType(16sc);
InstantiateInvokeAffineBackSrcP4ForGeomType(16sc);

} // namespace mpp::image::cuda
