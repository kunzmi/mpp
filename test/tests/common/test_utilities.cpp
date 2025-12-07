#include <catch2/catch_get_random_seed.hpp>
#include <catch2/catch_test_macros.hpp>
#include <climits>
#include <cmath>
#include <common/defines.h>
#include <common/numeric_limits.h>
#include <common/utilities.h>
#include <cstddef>
#include <cstdint>
#include <math.h>
#include <numeric>
#include <random>
#include <vector>

using namespace mpp;
using namespace Catch;

TEST_CASE("DivRoundUShort", "[Common]")
{
    using ComputeT = ushort;
    std::default_random_engine engine(Catch::getSeed());
    std::uniform_int_distribution<ComputeT> distri(0, 1000);

    size_t size = 1000;
    std::vector<double> input(size);
    std::vector<double> input2(size);

    for (size_t i = 0; i < size; i++)
    {
        input[i]  = static_cast<double>(distri(engine));
        input2[i] = static_cast<double>(distri(engine));
    }

    bool rnOK = true;
    bool azOK = true;
    bool rzOK = true;
    bool rdOK = true;
    bool ruOK = true;

    for (size_t i = 0; i < size; i++)
    {
        ComputeT n    = static_cast<ComputeT>(input[i]);
        ComputeT d    = static_cast<ComputeT>(input2[i]);
        double f_rn   = std::nearbyint(input[i] / input2[i]);
        ComputeT i_rn = DivRoundNearestEven(n, d);

        double f_az   = std::round(input[i] / input2[i]);
        ComputeT i_az = DivRoundTiesAwayFromZero(n, d);

        double f_rz   = std::trunc(input[i] / input2[i]);
        ComputeT i_rz = DivRoundTowardZero(n, d);

        double f_rd   = std::floor(input[i] / input2[i]);
        ComputeT i_rd = DivRoundTowardNegInf(n, d);

        double f_ru   = std::ceil(input[i] / input2[i]);
        ComputeT i_ru = DivRoundTowardPosInf(n, d);

        // handle INF
        if (std::isinf(f_rn))
        {
            f_rn = static_cast<double>(signbit(f_rn) ? mpp::numeric_limits<ComputeT>::lowest()
                                                     : mpp::numeric_limits<ComputeT>::max());
        }
        if (std::isinf(f_az))
        {
            f_az = static_cast<double>(signbit(f_az) ? mpp::numeric_limits<ComputeT>::lowest()
                                                     : mpp::numeric_limits<ComputeT>::max());
        }
        if (std::isinf(f_rz))
        {
            f_rz = static_cast<double>(signbit(f_rz) ? mpp::numeric_limits<ComputeT>::lowest()
                                                     : mpp::numeric_limits<ComputeT>::max());
        }
        if (std::isinf(f_rd))
        {
            f_rd = static_cast<double>(signbit(f_rd) ? mpp::numeric_limits<ComputeT>::lowest()
                                                     : mpp::numeric_limits<ComputeT>::max());
        }
        if (std::isinf(f_ru))
        {
            f_ru = static_cast<double>(signbit(f_ru) ? mpp::numeric_limits<ComputeT>::lowest()
                                                     : mpp::numeric_limits<ComputeT>::max());
        }

        rnOK &= f_rn == i_rn;
        azOK &= f_az == i_az;
        rzOK &= f_rz == i_rz;
        rdOK &= f_rd == i_rd;
        ruOK &= f_ru == i_ru;
    }

    CHECK(rnOK == true);
    CHECK(azOK == true);
    CHECK(rzOK == true);
    CHECK(rdOK == true);
    CHECK(ruOK == true);
}

TEST_CASE("DivRoundShort", "[Common]")
{
    using ComputeT = short;
    std::default_random_engine engine(Catch::getSeed());
    std::uniform_int_distribution<ComputeT> distri(-1000, 1000);

    size_t size = 1000;
    std::vector<double> input(size);
    std::vector<double> input2(size);

    for (size_t i = 0; i < size; i++)
    {
        input[i]  = static_cast<double>(distri(engine));
        input2[i] = static_cast<double>(distri(engine));
    }

    bool rnOK = true;
    bool azOK = true;
    bool rzOK = true;
    bool rdOK = true;
    bool ruOK = true;

    for (size_t i = 0; i < size; i++)
    {
        ComputeT n    = static_cast<ComputeT>(input[i]);
        ComputeT d    = static_cast<ComputeT>(input2[i]);
        double f_rn   = std::nearbyint(input[i] / input2[i]);
        ComputeT i_rn = DivRoundNearestEven(n, d);

        double f_az   = std::round(input[i] / input2[i]);
        ComputeT i_az = DivRoundTiesAwayFromZero(n, d);

        double f_rz   = std::trunc(input[i] / input2[i]);
        ComputeT i_rz = DivRoundTowardZero(n, d);

        double f_rd   = std::floor(input[i] / input2[i]);
        ComputeT i_rd = DivRoundTowardNegInf(n, d);

        double f_ru   = std::ceil(input[i] / input2[i]);
        ComputeT i_ru = DivRoundTowardPosInf(n, d);

        // handle INF
        if (std::isinf(f_rn))
        {
            f_rn = static_cast<double>(signbit(f_rn) ? mpp::numeric_limits<ComputeT>::lowest()
                                                     : mpp::numeric_limits<ComputeT>::max());
        }
        if (std::isinf(f_az))
        {
            f_az = static_cast<double>(signbit(f_az) ? mpp::numeric_limits<ComputeT>::lowest()
                                                     : mpp::numeric_limits<ComputeT>::max());
        }
        if (std::isinf(f_rz))
        {
            f_rz = static_cast<double>(signbit(f_rz) ? mpp::numeric_limits<ComputeT>::lowest()
                                                     : mpp::numeric_limits<ComputeT>::max());
        }
        if (std::isinf(f_rd))
        {
            f_rd = static_cast<double>(signbit(f_rd) ? mpp::numeric_limits<ComputeT>::lowest()
                                                     : mpp::numeric_limits<ComputeT>::max());
        }
        if (std::isinf(f_ru))
        {
            f_ru = static_cast<double>(signbit(f_ru) ? mpp::numeric_limits<ComputeT>::lowest()
                                                     : mpp::numeric_limits<ComputeT>::max());
        }

        rnOK &= f_rn == i_rn;
        azOK &= f_az == i_az;
        rzOK &= f_rz == i_rz;
        rdOK &= f_rd == i_rd;
        ruOK &= f_ru == i_ru;
    }

    CHECK(rnOK == true);
    CHECK(azOK == true);
    CHECK(rzOK == true);
    CHECK(rdOK == true);
    CHECK(ruOK == true);
}

TEST_CASE("DivRoundUInt", "[Common]")
{
    using ComputeT = uint;
    std::default_random_engine engine(Catch::getSeed());
    std::uniform_int_distribution<ComputeT> distri(0, 1000);

    size_t size = 1000;
    std::vector<double> input(size);
    std::vector<double> input2(size);

    for (size_t i = 0; i < size; i++)
    {
        input[i]  = static_cast<double>(distri(engine));
        input2[i] = static_cast<double>(distri(engine));
    }

    bool rnOK = true;
    bool azOK = true;
    bool rzOK = true;
    bool rdOK = true;
    bool ruOK = true;

    for (size_t i = 0; i < size; i++)
    {
        ComputeT n    = static_cast<ComputeT>(input[i]);
        ComputeT d    = static_cast<ComputeT>(input2[i]);
        double f_rn   = std::nearbyint(input[i] / input2[i]);
        ComputeT i_rn = DivRoundNearestEven(n, d);

        double f_az   = std::round(input[i] / input2[i]);
        ComputeT i_az = DivRoundTiesAwayFromZero(n, d);

        double f_rz   = std::trunc(input[i] / input2[i]);
        ComputeT i_rz = DivRoundTowardZero(n, d);

        double f_rd   = std::floor(input[i] / input2[i]);
        ComputeT i_rd = DivRoundTowardNegInf(n, d);

        double f_ru   = std::ceil(input[i] / input2[i]);
        ComputeT i_ru = DivRoundTowardPosInf(n, d);

        // handle INF
        if (std::isinf(f_rn))
        {
            f_rn = static_cast<double>(signbit(f_rn) ? mpp::numeric_limits<ComputeT>::lowest()
                                                     : mpp::numeric_limits<ComputeT>::max());
        }
        if (std::isinf(f_az))
        {
            f_az = static_cast<double>(signbit(f_az) ? mpp::numeric_limits<ComputeT>::lowest()
                                                     : mpp::numeric_limits<ComputeT>::max());
        }
        if (std::isinf(f_rz))
        {
            f_rz = static_cast<double>(signbit(f_rz) ? mpp::numeric_limits<ComputeT>::lowest()
                                                     : mpp::numeric_limits<ComputeT>::max());
        }
        if (std::isinf(f_rd))
        {
            f_rd = static_cast<double>(signbit(f_rd) ? mpp::numeric_limits<ComputeT>::lowest()
                                                     : mpp::numeric_limits<ComputeT>::max());
        }
        if (std::isinf(f_ru))
        {
            f_ru = static_cast<double>(signbit(f_ru) ? mpp::numeric_limits<ComputeT>::lowest()
                                                     : mpp::numeric_limits<ComputeT>::max());
        }

        rnOK &= f_rn == i_rn;
        azOK &= f_az == i_az;
        rzOK &= f_rz == i_rz;
        rdOK &= f_rd == i_rd;
        ruOK &= f_ru == i_ru;
    }

    CHECK(rnOK == true);
    CHECK(azOK == true);
    CHECK(rzOK == true);
    CHECK(rdOK == true);
    CHECK(ruOK == true);
}

TEST_CASE("DivRoundInt", "[Common]")
{
    using ComputeT = int;
    std::default_random_engine engine(Catch::getSeed());
    std::uniform_int_distribution<ComputeT> distri(-1000, 1000);

    size_t size = 1000;
    std::vector<double> input(size);
    std::vector<double> input2(size);

    for (size_t i = 0; i < size; i++)
    {
        input[i]  = static_cast<double>(distri(engine));
        input2[i] = static_cast<double>(distri(engine));
    }

    bool rnOK = true;
    bool azOK = true;
    bool rzOK = true;
    bool rdOK = true;
    bool ruOK = true;

    for (size_t i = 0; i < size; i++)
    {
        ComputeT n    = static_cast<ComputeT>(input[i]);
        ComputeT d    = static_cast<ComputeT>(input2[i]);
        double f_rn   = std::nearbyint(input[i] / input2[i]);
        ComputeT i_rn = DivRoundNearestEven(n, d);

        double f_az   = std::round(input[i] / input2[i]);
        ComputeT i_az = DivRoundTiesAwayFromZero(n, d);

        double f_rz   = std::trunc(input[i] / input2[i]);
        ComputeT i_rz = DivRoundTowardZero(n, d);

        double f_rd   = std::floor(input[i] / input2[i]);
        ComputeT i_rd = DivRoundTowardNegInf(n, d);

        double f_ru   = std::ceil(input[i] / input2[i]);
        ComputeT i_ru = DivRoundTowardPosInf(n, d);

        // handle INF
        if (std::isinf(f_rn))
        {
            f_rn = static_cast<double>(signbit(f_rn) ? mpp::numeric_limits<ComputeT>::lowest()
                                                     : mpp::numeric_limits<ComputeT>::max());
        }
        if (std::isinf(f_az))
        {
            f_az = static_cast<double>(signbit(f_az) ? mpp::numeric_limits<ComputeT>::lowest()
                                                     : mpp::numeric_limits<ComputeT>::max());
        }
        if (std::isinf(f_rz))
        {
            f_rz = static_cast<double>(signbit(f_rz) ? mpp::numeric_limits<ComputeT>::lowest()
                                                     : mpp::numeric_limits<ComputeT>::max());
        }
        if (std::isinf(f_rd))
        {
            f_rd = static_cast<double>(signbit(f_rd) ? mpp::numeric_limits<ComputeT>::lowest()
                                                     : mpp::numeric_limits<ComputeT>::max());
        }
        if (std::isinf(f_ru))
        {
            f_ru = static_cast<double>(signbit(f_ru) ? mpp::numeric_limits<ComputeT>::lowest()
                                                     : mpp::numeric_limits<ComputeT>::max());
        }

        rnOK &= f_rn == i_rn;
        azOK &= f_az == i_az;
        rzOK &= f_rz == i_rz;
        rdOK &= f_rd == i_rd;
        ruOK &= f_ru == i_ru;
    }

    CHECK(rnOK == true);
    CHECK(azOK == true);
    CHECK(rzOK == true);
    CHECK(rdOK == true);
    CHECK(ruOK == true);
}

TEST_CASE("DivRoundULong", "[Common]")
{
    using ComputeT = ulong64;
    std::default_random_engine engine(Catch::getSeed());
    std::uniform_int_distribution<ComputeT> distri(0, 1000);

    size_t size = 1000;
    std::vector<double> input(size);
    std::vector<double> input2(size);

    for (size_t i = 0; i < size; i++)
    {
        input[i]  = static_cast<double>(distri(engine));
        input2[i] = static_cast<double>(distri(engine));
    }

    bool rnOK = true;
    bool azOK = true;
    bool rzOK = true;
    bool rdOK = true;
    bool ruOK = true;

    for (size_t i = 0; i < size; i++)
    {
        ComputeT n    = static_cast<ComputeT>(input[i]);
        ComputeT d    = static_cast<ComputeT>(input2[i]);
        double f_rn   = std::nearbyint(input[i] / input2[i]);
        ComputeT i_rn = DivRoundNearestEven(n, d);

        double f_az   = std::round(input[i] / input2[i]);
        ComputeT i_az = DivRoundTiesAwayFromZero(n, d);

        double f_rz   = std::trunc(input[i] / input2[i]);
        ComputeT i_rz = DivRoundTowardZero(n, d);

        double f_rd   = std::floor(input[i] / input2[i]);
        ComputeT i_rd = DivRoundTowardNegInf(n, d);

        double f_ru   = std::ceil(input[i] / input2[i]);
        ComputeT i_ru = DivRoundTowardPosInf(n, d);

        // handle INF
        if (std::isinf(f_rn))
        {
            f_rn = static_cast<double>(signbit(f_rn) ? mpp::numeric_limits<ComputeT>::lowest()
                                                     : mpp::numeric_limits<ComputeT>::max());
        }
        if (std::isinf(f_az))
        {
            f_az = static_cast<double>(signbit(f_az) ? mpp::numeric_limits<ComputeT>::lowest()
                                                     : mpp::numeric_limits<ComputeT>::max());
        }
        if (std::isinf(f_rz))
        {
            f_rz = static_cast<double>(signbit(f_rz) ? mpp::numeric_limits<ComputeT>::lowest()
                                                     : mpp::numeric_limits<ComputeT>::max());
        }
        if (std::isinf(f_rd))
        {
            f_rd = static_cast<double>(signbit(f_rd) ? mpp::numeric_limits<ComputeT>::lowest()
                                                     : mpp::numeric_limits<ComputeT>::max());
        }
        if (std::isinf(f_ru))
        {
            f_ru = static_cast<double>(signbit(f_ru) ? mpp::numeric_limits<ComputeT>::lowest()
                                                     : mpp::numeric_limits<ComputeT>::max());
        }

        rnOK &= f_rn == i_rn;
        azOK &= f_az == i_az;
        rzOK &= f_rz == i_rz;
        rdOK &= f_rd == i_rd;
        ruOK &= f_ru == i_ru;
    }

    CHECK(rnOK == true);
    CHECK(azOK == true);
    CHECK(rzOK == true);
    CHECK(rdOK == true);
    CHECK(ruOK == true);
}

TEST_CASE("DivRoundLong", "[Common]")
{
    using ComputeT = long64;
    std::default_random_engine engine(Catch::getSeed());
    std::uniform_int_distribution<ComputeT> distri(-1000, 1000);

    size_t size = 1000;
    std::vector<double> input(size);
    std::vector<double> input2(size);

    for (size_t i = 0; i < size; i++)
    {
        input[i]  = static_cast<double>(distri(engine));
        input2[i] = static_cast<double>(distri(engine));
    }

    bool rnOK = true;
    bool azOK = true;
    bool rzOK = true;
    bool rdOK = true;
    bool ruOK = true;

    for (size_t i = 0; i < size; i++)
    {
        ComputeT n    = static_cast<ComputeT>(input[i]);
        ComputeT d    = static_cast<ComputeT>(input2[i]);
        double f_rn   = std::nearbyint(input[i] / input2[i]);
        ComputeT i_rn = DivRoundNearestEven(n, d);

        double f_az   = std::round(input[i] / input2[i]);
        ComputeT i_az = DivRoundTiesAwayFromZero(n, d);

        double f_rz   = std::trunc(input[i] / input2[i]);
        ComputeT i_rz = DivRoundTowardZero(n, d);

        double f_rd   = std::floor(input[i] / input2[i]);
        ComputeT i_rd = DivRoundTowardNegInf(n, d);

        double f_ru   = std::ceil(input[i] / input2[i]);
        ComputeT i_ru = DivRoundTowardPosInf(n, d);

        // handle INF
        if (std::isinf(f_rn))
        {
            f_rn = static_cast<double>(signbit(f_rn) ? mpp::numeric_limits<ComputeT>::lowest()
                                                     : mpp::numeric_limits<ComputeT>::max());
        }
        if (std::isinf(f_az))
        {
            f_az = static_cast<double>(signbit(f_az) ? mpp::numeric_limits<ComputeT>::lowest()
                                                     : mpp::numeric_limits<ComputeT>::max());
        }
        if (std::isinf(f_rz))
        {
            f_rz = static_cast<double>(signbit(f_rz) ? mpp::numeric_limits<ComputeT>::lowest()
                                                     : mpp::numeric_limits<ComputeT>::max());
        }
        if (std::isinf(f_rd))
        {
            f_rd = static_cast<double>(signbit(f_rd) ? mpp::numeric_limits<ComputeT>::lowest()
                                                     : mpp::numeric_limits<ComputeT>::max());
        }
        if (std::isinf(f_ru))
        {
            f_ru = static_cast<double>(signbit(f_ru) ? mpp::numeric_limits<ComputeT>::lowest()
                                                     : mpp::numeric_limits<ComputeT>::max());
        }

        rnOK &= f_rn == i_rn;
        azOK &= f_az == i_az;
        rzOK &= f_rz == i_rz;
        rdOK &= f_rd == i_rd;
        ruOK &= f_ru == i_ru;
    }

    CHECK(rnOK == true);
    CHECK(azOK == true);
    CHECK(rzOK == true);
    CHECK(rdOK == true);
    CHECK(ruOK == true);
}

TEST_CASE("DivScaleRoundUShort", "[Common]")
{
    using ComputeT = ushort;
    std::default_random_engine engine(Catch::getSeed());
    std::uniform_int_distribution<ComputeT> distri(0, 1000);
    std::uniform_int_distribution<ComputeT> distri2(1, 64);

    size_t size = 1000;
    std::vector<double> input(size);
    std::vector<double> input2(size);

    for (size_t i = 0; i < size; i++)
    {
        input[i]  = static_cast<double>(distri(engine));
        input2[i] = static_cast<double>(distri2(engine));
    }

    bool rnOK = true;

    for (size_t i = 0; i < size; i++)
    {
        ComputeT n    = static_cast<ComputeT>(input[i]);
        ComputeT d    = static_cast<ComputeT>(input2[i]);
        double f_rn   = std::nearbyint(input[i] / input2[i]);
        ComputeT i_rn = DivScaleRoundNearestEven(n, d);

        rnOK &= f_rn == i_rn;
    }

    CHECK(rnOK == true);
}

TEST_CASE("DivScaleRoundShort", "[Common]")
{
    using ComputeT = short;
    std::default_random_engine engine(Catch::getSeed());
    std::uniform_int_distribution<ComputeT> distri(-1000, 1000);
    std::uniform_int_distribution<ComputeT> distri2(1, 64);

    size_t size = 1000;
    std::vector<double> input(size);
    std::vector<double> input2(size);

    for (size_t i = 0; i < size; i++)
    {
        input[i]  = static_cast<double>(distri(engine));
        input2[i] = static_cast<double>(distri2(engine));
    }

    bool rnOK = true;

    for (size_t i = 0; i < size; i++)
    {
        ComputeT n    = static_cast<ComputeT>(input[i]);
        ComputeT d    = static_cast<ComputeT>(input2[i]);
        double f_rn   = std::nearbyint(input[i] / input2[i]);
        ComputeT i_rn = DivScaleRoundNearestEven(n, d);

        rnOK &= f_rn == i_rn;
    }

    CHECK(rnOK == true);
}

TEST_CASE("DivScaleRoundUInt", "[Common]")
{
    using ComputeT = uint;
    std::default_random_engine engine(Catch::getSeed());
    std::uniform_int_distribution<ComputeT> distri(0, 1000);
    std::uniform_int_distribution<ComputeT> distri2(1, 64);

    size_t size = 1000;
    std::vector<double> input(size);
    std::vector<double> input2(size);

    for (size_t i = 0; i < size; i++)
    {
        input[i]  = static_cast<double>(distri(engine));
        input2[i] = static_cast<double>(distri2(engine));
    }

    bool rnOK = true;

    for (size_t i = 0; i < size; i++)
    {
        ComputeT n    = static_cast<ComputeT>(input[i]);
        ComputeT d    = static_cast<ComputeT>(input2[i]);
        double f_rn   = std::nearbyint(input[i] / input2[i]);
        ComputeT i_rn = DivScaleRoundNearestEven(n, d);

        rnOK &= f_rn == i_rn;
    }

    CHECK(rnOK == true);
}

TEST_CASE("DivScaleRoundInt", "[Common]")
{
    using ComputeT = int;
    std::default_random_engine engine(Catch::getSeed());
    std::uniform_int_distribution<ComputeT> distri(-1000, 1000);
    std::uniform_int_distribution<ComputeT> distri2(1, 64);

    size_t size = 1000;
    std::vector<double> input(size);
    std::vector<double> input2(size);

    for (size_t i = 0; i < size; i++)
    {
        input[i]  = static_cast<double>(distri(engine));
        input2[i] = static_cast<double>(distri2(engine));
    }

    bool rnOK = true;

    for (size_t i = 0; i < size; i++)
    {
        ComputeT n    = static_cast<ComputeT>(input[i]);
        ComputeT d    = static_cast<ComputeT>(input2[i]);
        double f_rn   = std::nearbyint(input[i] / input2[i]);
        ComputeT i_rn = DivScaleRoundNearestEven(n, d);

        rnOK &= f_rn == i_rn;
    }

    CHECK(rnOK == true);
}

TEST_CASE("DivScaleRoundULong", "[Common]")
{
    using ComputeT = ulong64;
    std::default_random_engine engine(Catch::getSeed());
    std::uniform_int_distribution<ComputeT> distri(0, 1000);
    std::uniform_int_distribution<ComputeT> distri2(1, 64);

    size_t size = 1000;
    std::vector<double> input(size);
    std::vector<double> input2(size);

    for (size_t i = 0; i < size; i++)
    {
        input[i]  = static_cast<double>(distri(engine));
        input2[i] = static_cast<double>(distri2(engine));
    }

    bool rnOK = true;

    for (size_t i = 0; i < size; i++)
    {
        ComputeT n    = static_cast<ComputeT>(input[i]);
        ComputeT d    = static_cast<ComputeT>(input2[i]);
        double f_rn   = std::nearbyint(input[i] / input2[i]);
        ComputeT i_rn = DivScaleRoundNearestEven(n, d);

        rnOK &= f_rn == i_rn;
    }

    CHECK(rnOK == true);
}

TEST_CASE("DivScaleRoundLong", "[Common]")
{
    using ComputeT = long64;
    std::default_random_engine engine(Catch::getSeed());
    std::uniform_int_distribution<ComputeT> distri(-1000, 1000);
    std::uniform_int_distribution<ComputeT> distri2(1, 64);

    size_t size = 1000;
    std::vector<double> input(size);
    std::vector<double> input2(size);

    for (size_t i = 0; i < size; i++)
    {
        input[i]  = static_cast<double>(distri(engine));
        input2[i] = static_cast<double>(distri2(engine));
    }

    bool rnOK = true;

    for (size_t i = 0; i < size; i++)
    {
        ComputeT n    = static_cast<ComputeT>(input[i]);
        ComputeT d    = static_cast<ComputeT>(input2[i]);
        double f_rn   = std::nearbyint(input[i] / input2[i]);
        ComputeT i_rn = DivScaleRoundNearestEven(n, d);

        rnOK &= f_rn == i_rn;
    }

    CHECK(rnOK == true);
}