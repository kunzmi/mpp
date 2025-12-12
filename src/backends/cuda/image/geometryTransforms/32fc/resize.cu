#include "../resize_impl.h"

using namespace mpp::cuda;

namespace mpp::image::cuda
{

ForAllChannelsNoAlpha(32fc);
InstantiateP2_ForGeomType(32fc);
InstantiateP3_ForGeomType(32fc);
InstantiateP4_ForGeomType(32fc);

} // namespace mpp::image::cuda
