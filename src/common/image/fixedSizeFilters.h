#pragma once
#include <common/defines.h>
#include <common/image/pixelTypes.h>
#include <common/numberTypes.h>
#include <common/opp_defs.h>

namespace opp::image
{
// NVCC / Cuda cannot access the values defined as MYCONST from device code. To circumvent that restriction,
// we'll define the filter matrices as const members for NVCC and as MYCONST for host compilers...

#ifdef IS_CUDA_COMPILER
#define MYCONST const
#else
#define MYCONST static constexpr
#endif

template <opp::FixedFilter filter, int filterSize, typename filterT> struct FixedFilterKernel
{
};

template <> struct FixedFilterKernel<opp::FixedFilter::Gauss, 3, float>
{
    MYCONST float Values[3][3]       = {{0.0751136106612534f, 0.123841402677493f, 0.0751136106612534f},
                                        {0.123841402677493f, 0.204179946645014f, 0.123841402677493f},
                                        {0.0751136106612534f, 0.123841402677493f, 0.0751136106612534f}};
    MYCONST float ValuesSeparable[3] = {0.274068624f, 0.451862752f, 0.274068624f};
    MYCONST float Scaling            = 1;

    MYCONST bool NeedsScaling = false;
};
template <> struct FixedFilterKernel<opp::FixedFilter::Gauss, 5, float>
{
    MYCONST float Values[5][5] = {
        {0.0121461294818704f, 0.0261099496482738f, 0.0336973250459961f, 0.0261099496482738f, 0.0121461294818704f},
        {0.0261099496482738f, 0.0561273014298881f, 0.0724375169510366f, 0.0561273014298881f, 0.0261099496482738f},
        {0.0336973250459961f, 0.0724375169510366f, 0.0934873711786461f, 0.0724375169510366f, 0.0336973250459961f},
        {0.0261099496482738f, 0.0561273014298881f, 0.0724375169510366f, 0.0561273014298881f, 0.0261099496482738f},
        {0.0121461294818704f, 0.0261099496482738f, 0.0336973250459961f, 0.0261099496482738f, 0.0121461294818704f}};
    MYCONST float ValuesSeparable[5] = {0.11020948f, 0.236912012f, 0.305757046f, 0.236912012f, 0.11020948f};
    MYCONST float Scaling            = 1;

    MYCONST bool NeedsScaling = false;
};
template <> struct FixedFilterKernel<opp::FixedFilter::Gauss, 7, float>
{
    MYCONST float ValuesSeparable[7] = {0.0581035763f, 0.125691682f, 0.199695736f, 0.233018041f,
                                        0.199695736f,  0.125691682f, 0.0581035763};
    MYCONST float Scaling            = 1;

    MYCONST bool NeedsScaling = false;
};
template <> struct FixedFilterKernel<opp::FixedFilter::Gauss, 9, float>
{
    MYCONST float ValuesSeparable[9] = {0.0361378156f, 0.0744762495f, 0.124837041f,  0.170191884f, 0.188713938f,
                                        0.170191884f,  0.124837041f,  0.0744762495f, 0.0361378156f};
    MYCONST float Scaling            = 1;

    MYCONST bool NeedsScaling = false;
};
template <> struct FixedFilterKernel<opp::FixedFilter::Gauss, 11, float>
{
    MYCONST float ValuesSeparable[11] = {0.0249794535f, 0.048605375f, 0.0815719739f, 0.118073598f,
                                         0.147407621f,  0.158723891f, 0.147407621f,  0.118073598f,
                                         0.0815719739f, 0.048605375f, 0.0249794535f};
    MYCONST float Scaling             = 1;

    MYCONST bool NeedsScaling = false;
};
template <> struct FixedFilterKernel<opp::FixedFilter::Gauss, 13, float>
{
    MYCONST float ValuesSeparable[13] = {0.0185440257f, 0.034166947f, 0.0563317686f, 0.0831085443f, 0.109719299f,
                                         0.129618034f,  0.137022823f, 0.129618034f,  0.109719299f,  0.0831085443f,
                                         0.0563317686f, 0.034166947f, 0.0185440257f};
    MYCONST float Scaling             = 1;

    MYCONST bool NeedsScaling = false;
};
template <> struct FixedFilterKernel<opp::FixedFilter::Gauss, 15, float>
{
    MYCONST float ValuesSeparable[15] = {0.014481457f,  0.0254102573f, 0.0408918671f, 0.0603526309f, 0.0816933662f,
                                         0.101416491f,  0.115468003f,  0.120571882f,  0.115468003f,  0.101416491f,
                                         0.0816933662f, 0.0603526309f, 0.0408918671f, 0.0254102573f, 0.014481457f};
    MYCONST float Scaling             = 1;

    MYCONST bool NeedsScaling = false;
};

template <typename filterT> struct FixedFilterKernel<opp::FixedFilter::HighPass, 3, filterT>
{
    MYCONST filterT Values[3][3] = {{-1, -1, -1}, {-1, 8, -1}, {-1, -1, -1}};
    MYCONST filterT Scaling      = 1;

    MYCONST bool NeedsScaling = false;
};

template <> struct FixedFilterKernel<opp::FixedFilter::LowPass, 3, int>
{
    MYCONST int Values[3][3] = {{1, 1, 1}, {1, 1, 1}, {1, 1, 1}};
    MYCONST int Scaling      = 9;

    MYCONST bool NeedsScaling = true;
};
template <> struct FixedFilterKernel<opp::FixedFilter::LowPass, 3, float>
{
    MYCONST float Values[3][3] = {
        {1 / 9.0f, 1 / 9.0f, 1 / 9.0f}, {1 / 9.0f, 1 / 9.0f, 1 / 9.0f}, {1 / 9.0f, 1 / 9.0f, 1 / 9.0f}};

    MYCONST bool NeedsScaling = false;
};

template <typename filterT> struct FixedFilterKernel<opp::FixedFilter::Laplace, 3, filterT>
{
    MYCONST filterT Values[3][3] = {{-1, -1, -1}, {-1, 8, -1}, {-1, -1, -1}};
    MYCONST filterT Scaling      = 1;

    MYCONST bool NeedsScaling = false;
};

template <typename filterT> struct FixedFilterKernel<opp::FixedFilter::PrewittHoriz, 3, filterT>
{
    MYCONST filterT Values[3][3] = {{1, 1, 1}, {0, 0, 0}, {-1, -1, -1}};
    MYCONST filterT Scaling      = 1;

    MYCONST bool NeedsScaling = false;
};

template <typename filterT> struct FixedFilterKernel<opp::FixedFilter::PrewittVert, 3, filterT>
{
    MYCONST filterT Values[3][3] = {{-1, 0, 1}, {-1, 0, 1}, {-1, 0, 1}};
    MYCONST filterT Scaling      = 1;

    MYCONST bool NeedsScaling = false;
};

template <typename filterT> struct FixedFilterKernel<opp::FixedFilter::RobertsDown, 3, filterT>
{
    MYCONST filterT Values[3][3] = {{0, 0, 0}, {0, 1, 0}, {0, 0, -1}};
    MYCONST filterT Scaling      = 1;

    MYCONST bool NeedsScaling = false;
};

template <typename filterT> struct FixedFilterKernel<opp::FixedFilter::RobertsUp, 3, filterT>
{
    MYCONST filterT Values[3][3] = {{0, 0, 0}, {0, 1, 0}, {-1, 0, 0}};
    MYCONST filterT Scaling      = 1;

    MYCONST bool NeedsScaling = false;
};

template <typename filterT> struct FixedFilterKernel<opp::FixedFilter::ScharrHoriz, 3, filterT>
{
    MYCONST filterT Values[3][3] = {{3, 10, 3}, {0, 0, 0}, {-3, -10, -3}};
    MYCONST filterT Scaling      = 1;

    MYCONST bool NeedsScaling = false;
};

template <typename filterT> struct FixedFilterKernel<opp::FixedFilter::ScharrVert, 3, filterT>
{
    MYCONST filterT Values[3][3] = {{-3, 0, 3}, {-10, 0, 10}, {-3, 0, 3}};
    MYCONST filterT Scaling      = 1;

    MYCONST bool NeedsScaling = false;
};

template <> struct FixedFilterKernel<opp::FixedFilter::Sharpen, 3, int>
{
    MYCONST int Values[3][3] = {{-1, -1, -1}, {-1, 16, -1}, {-1, -1, -1}};
    MYCONST int Scaling      = 8;

    MYCONST bool NeedsScaling = true;
};
template <> struct FixedFilterKernel<opp::FixedFilter::Sharpen, 3, float>
{
    MYCONST float Values[3][3] = {
        {-1 / 8.0f, -1 / 8.0f, -1 / 8.0f}, {-1 / 8.0f, 16 / 8.0f, -1 / 8.0f}, {-1 / 8.0f, -1 / 8.0f, -1 / 8.0f}};

    MYCONST bool NeedsScaling = false;
};

template <typename filterT> struct FixedFilterKernel<opp::FixedFilter::SobelCross, 3, filterT>
{
    MYCONST filterT Values[3][3] = {{-1, 0, 1}, {0, 0, 0}, {1, 0, -1}};
    MYCONST filterT Scaling      = 1;

    MYCONST bool NeedsScaling = false;
};

template <typename filterT> struct FixedFilterKernel<opp::FixedFilter::SobelHoriz, 3, filterT>
{
    MYCONST filterT Values[3][3] = {{1, 2, 1}, {0, 0, 0}, {-1, -2, -1}};
    MYCONST filterT Scaling      = 1;

    MYCONST bool NeedsScaling = false;
};

template <typename filterT> struct FixedFilterKernel<opp::FixedFilter::SobelVert, 3, filterT>
{
    MYCONST filterT Values[3][3] = {{-1, 0, 1}, {-2, 0, 2}, {-1, 0, 1}};
    MYCONST filterT Scaling      = 1;

    MYCONST bool NeedsScaling = false;
};

template <typename filterT> struct FixedFilterKernel<opp::FixedFilter::SobelHorizSecond, 3, filterT>
{
    MYCONST filterT Values[3][3] = {{1, 2, 1}, {-2, -4, -2}, {1, 2, 1}};
    MYCONST filterT Scaling      = 1;

    MYCONST bool NeedsScaling = false;
};

template <typename filterT> struct FixedFilterKernel<opp::FixedFilter::SobelVertSecond, 3, filterT>
{
    MYCONST filterT Values[3][3] = {{1, -2, 1}, {2, 4, 2}, {1, -2, 1}};
    MYCONST filterT Scaling      = 1;

    MYCONST bool NeedsScaling = false;
};

template <typename filterT> struct FixedFilterKernel<opp::FixedFilter::HighPass, 5, filterT>
{
    MYCONST filterT Values[5][5] = {{-1, -1, -1, -1, -1}, //
                                    {-1, -1, -1, -1, -1}, //
                                    {-1, -1, 24, -1, -1}, //
                                    {-1, -1, -1, -1, -1}, //
                                    {-1, -1, -1, -1, -1}};
    MYCONST filterT Scaling      = 1;

    MYCONST bool NeedsScaling = false;
};

template <> struct FixedFilterKernel<opp::FixedFilter::LowPass, 5, int>
{
    MYCONST int Values[5][5] = {{1, 1, 1, 1, 1}, //
                                {1, 1, 1, 1, 1}, //
                                {1, 1, 1, 1, 1}, //
                                {1, 1, 1, 1, 1}, //
                                {1, 1, 1, 1, 1}};
    MYCONST int Scaling      = 25;

    MYCONST bool NeedsScaling = false;
};
template <> struct FixedFilterKernel<opp::FixedFilter::LowPass, 5, float>
{
    MYCONST float Values[5][5] = {{1 / 25.0f, 1 / 25.0f, 1 / 25.0f, 1 / 25.0f, 1 / 25.0f}, //
                                  {1 / 25.0f, 1 / 25.0f, 1 / 25.0f, 1 / 25.0f, 1 / 25.0f}, //
                                  {1 / 25.0f, 1 / 25.0f, 1 / 25.0f, 1 / 25.0f, 1 / 25.0f}, //
                                  {1 / 25.0f, 1 / 25.0f, 1 / 25.0f, 1 / 25.0f, 1 / 25.0f}, //
                                  {1 / 25.0f, 1 / 25.0f, 1 / 25.0f, 1 / 25.0f, 1 / 25.0f}};

    MYCONST bool NeedsScaling = false;
};

template <typename filterT> struct FixedFilterKernel<opp::FixedFilter::Laplace, 5, filterT>
{
    MYCONST filterT Values[5][5] = {{-1, -3, -4, -3, -1}, //
                                    {-3, 0, 6, 0, -3},    //
                                    {-4, 6, 20, 6, -4},   //
                                    {-3, 0, 6, 0, -3},    //
                                    {-1, -3, -4, -3, -1}};
    MYCONST filterT Scaling      = 1;

    MYCONST bool NeedsScaling = false;
};

template <typename filterT> struct FixedFilterKernel<opp::FixedFilter::SobelCross, 5, filterT>
{
    MYCONST filterT Values[5][5] = {{-1, -2, 0, 2, 1}, //
                                    {-2, -4, 0, 4, 2}, //
                                    {0, 0, 0, 0, 0},   //
                                    {2, 4, 0, -4, -2}, //
                                    {1, 2, 0, -2, -1}};
    MYCONST filterT Scaling      = 1;

    MYCONST bool NeedsScaling = false;
};

template <typename filterT> struct FixedFilterKernel<opp::FixedFilter::SobelHoriz, 5, filterT>
{
    MYCONST filterT Values[5][5] = {{1, 4, 6, 4, 1},       //
                                    {2, 8, 12, 18, 2},     //
                                    {0, 0, 0, 0, 0},       //
                                    {-2, -8, -12, -8, -2}, //
                                    {-1, -4, -6, -4, -1}};
    MYCONST filterT Scaling      = 1;

    MYCONST bool NeedsScaling = false;
};

template <typename filterT> struct FixedFilterKernel<opp::FixedFilter::SobelVert, 5, filterT>
{
    MYCONST filterT Values[5][5] = {{-1, -2, 0, 2, 1},   //
                                    {-4, -8, 0, 8, 4},   //
                                    {-6, -12, 0, 12, 6}, //
                                    {-4, -8, 0, 8, 4},   //
                                    {-1, -2, 0, 2, 1}};
    MYCONST filterT Scaling      = 1;

    MYCONST bool NeedsScaling = false;
};

template <typename filterT> struct FixedFilterKernel<opp::FixedFilter::SobelHorizSecond, 5, filterT>
{
    MYCONST filterT Values[5][5] = {{1, 4, 6, 4, 1},        //
                                    {0, 0, 0, 0, 0},        //
                                    {-2, -8, -12, -18, -2}, //
                                    {0, 0, 0, 0, 0},        //
                                    {1, 4, 6, 4, 1}};
    MYCONST filterT Scaling      = 1;

    MYCONST bool NeedsScaling = false;
};

template <typename filterT> struct FixedFilterKernel<opp::FixedFilter::SobelVertSecond, 5, filterT>
{
    MYCONST filterT Values[5][5] = {{1, 0, -2, 0, 1},  //
                                    {4, 0, -8, 0, 4},  //
                                    {6, 0, -12, 0, 6}, //
                                    {4, 0, -8, 0, 4},  //
                                    {1, 0, -2, 0, 1}};
    MYCONST filterT Scaling      = 1;

    MYCONST bool NeedsScaling = false;
};

#undef MYCONST
} // namespace opp::image