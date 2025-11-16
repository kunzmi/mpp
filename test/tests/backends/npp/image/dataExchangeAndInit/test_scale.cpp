#include <backends/npp/image/image16s.h>
#include <backends/npp/image/image16sC1View.h>
#include <backends/npp/image/image16sC2View.h>
#include <backends/npp/image/image16sC3View.h>
#include <backends/npp/image/image16sC4View.h>
#include <backends/npp/image/image16u.h>
#include <backends/npp/image/image16uC1View.h>
#include <backends/npp/image/image16uC2View.h>
#include <backends/npp/image/image16uC3View.h>
#include <backends/npp/image/image16uC4View.h>
#include <backends/npp/image/image32f.h>
#include <backends/npp/image/image32fC1View.h>
#include <backends/npp/image/image32fC2View.h>
#include <backends/npp/image/image32fC3View.h>
#include <backends/npp/image/image32fC4View.h>
#include <backends/npp/image/image32s.h>
#include <backends/npp/image/image32sC1View.h>
#include <backends/npp/image/image32sC2View.h>
#include <backends/npp/image/image32sC3View.h>
#include <backends/npp/image/image32sC4View.h>
#include <backends/npp/image/image32u.h>
#include <backends/npp/image/image32uC1View.h>
#include <backends/npp/image/image32uC2View.h>
#include <backends/npp/image/image32uC3View.h>
#include <backends/npp/image/image32uC4View.h>
#include <backends/npp/image/image64f.h>
#include <backends/npp/image/image64fC1View.h>
#include <backends/npp/image/image64fC2View.h>
#include <backends/npp/image/image64fC3View.h>
#include <backends/npp/image/image64fC4View.h>
#include <backends/npp/image/image8s.h>
#include <backends/npp/image/image8sC1View.h>
#include <backends/npp/image/image8sC2View.h>
#include <backends/npp/image/image8sC3View.h>
#include <backends/npp/image/image8sC4View.h>
#include <backends/npp/image/image8u.h>
#include <backends/npp/image/image8uC1View.h>
#include <backends/npp/image/image8uC2View.h>
#include <backends/npp/image/image8uC3View.h>
#include <backends/npp/image/image8uC4View.h>
#include <backends/simple_cpu/image/image.h>
#include <backends/simple_cpu/image/imageView.h>
#include <backends/simple_cpu/operator_random.h>
#include <catch2/catch_approx.hpp>
#include <catch2/catch_get_random_seed.hpp>
#include <catch2/catch_test_macros.hpp>
#include <common/defines.h>

using namespace mpp;
using namespace mpp::image;
using namespace Catch;
namespace cpu = mpp::image::cpuSimple;
namespace nv  = mpp::image::npp;

constexpr int size = 128;

TEST_CASE("8uC1", "[NPP.DataExchangeAndInit.Scale]")
{
    const uint seed         = Catch::getSeed();
    NppStreamContext nppCtx = nv::Image8uC1::GetStreamContext();

    cpu::Image<Pixel8uC1> cpu_src1(size, size);
    nv::Image8uC1 npp_src1(size, size);
    cpu::Image<Pixel32fC1> cpu_dst32f(size, size);
    cpu::Image<Pixel32fC1> npp_res32f(size, size);
    nv::Image32fC1 npp_dst32f(size, size);
    cpu::Image<Pixel16sC1> cpu_dst16s(size, size);
    cpu::Image<Pixel16sC1> npp_res16s(size, size);
    nv::Image16sC1 npp_dst16s(size, size);
    cpu::Image<Pixel16uC1> cpu_dst16u(size, size);
    cpu::Image<Pixel16uC1> npp_res16u(size, size);
    nv::Image16uC1 npp_dst16u(size, size);

    cpu_src1.FillRandom(seed);

    cpu_src1 >> npp_src1;

    npp_src1.Scale(npp_dst32f, 2.0f, 200.0f, nppCtx);
    npp_res32f << npp_dst32f;

    cpu_src1.Scale(cpu_dst32f, 2.0f, 200.0f);

    CHECK(cpu_dst32f.IsSimilar(npp_res32f, 0.0001f));

    npp_src1.Scale(npp_dst16u, nppCtx);
    npp_res16u << npp_dst16u;

    cpu_src1.Scale(cpu_dst16u);

    CHECK(cpu_dst16u.IsIdentical(npp_res16u));

    npp_src1.Scale(npp_dst16s, nppCtx);
    npp_res16s << npp_dst16s;

    cpu_src1.Scale(cpu_dst16s);

    CHECK(cpu_dst16s.IsIdentical(npp_res16s));
}