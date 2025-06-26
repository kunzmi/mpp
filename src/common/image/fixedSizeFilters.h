#pragma once
#include <common/defines.h>
#include <common/image/pixelTypes.h>
#include <common/numberTypes.h>
#include <common/mpp_defs.h>
#include <numbers>

namespace mpp::image
{
// NVCC / Cuda cannot access the values defined as "static constexpr" from device code. To circumvent that restriction,
// we'll define the filter matrices as const members for NVCC and as static constexpr for all other compilers...

#ifdef IS_CUDA_COMPILER
#define MYCONST const
#else
#define MYCONST static constexpr
#endif

template <mpp::FixedFilter filter, int filterSize, typename filterT> struct FixedFilterKernel
{
};
template <mpp::FixedFilter filter, int filterSize, typename filterT> struct FixedInvertedFilterKernel
{
    // GradientVector functions use an inverted definition of the SobelVert and PrewittVert filter kernel. In order to
    // stay compatible with NPP (and likely IPP) we define them inverted, too.
};

// Special case for SSIM using a 11x11 Gauss filter with sigma 1.5:
struct FixedFilterKernelSSIM
{
    MYCONST float ValuesSeparable[11] = {0.001028380084479f, 0.007598758135239f, 0.036000772128431f, 0.109360689509700f,
                                         0.213005537711254f, 0.266011724861794f, 0.213005537711254f, 0.109360689509700f,
                                         0.036000772128431f, 0.007598758135239f, 0.001028380084479f};
    MYCONST float Scaling             = 1;

    MYCONST bool NeedsScaling = false;
};

template <> struct FixedFilterKernel<mpp::FixedFilter::Gauss, 3, float>
{
    MYCONST float Values[3][3]       = {{0.0751136106612534f, 0.123841402677493f, 0.0751136106612534f},
                                        {0.123841402677493f, 0.204179946645014f, 0.123841402677493f},
                                        {0.0751136106612534f, 0.123841402677493f, 0.0751136106612534f}};
    MYCONST float ValuesSeparable[3] = {0.274068624f, 0.451862752f, 0.274068624f};
    MYCONST float Scaling            = 1;

    MYCONST bool NeedsScaling = false;
};
template <> struct FixedFilterKernel<mpp::FixedFilter::Gauss, 5, float>
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
template <> struct FixedFilterKernel<mpp::FixedFilter::Gauss, 7, float>
{
    MYCONST float ValuesSeparable[7] = {0.0581035763f, 0.125691682f, 0.199695736f, 0.233018041f,
                                        0.199695736f,  0.125691682f, 0.0581035763f};
    MYCONST float Scaling            = 1;

    MYCONST bool NeedsScaling = false;
};
template <> struct FixedFilterKernel<mpp::FixedFilter::Gauss, 9, float>
{
    MYCONST float ValuesSeparable[9] = {0.0361378156f, 0.0744762495f, 0.124837041f,  0.170191884f, 0.188713938f,
                                        0.170191884f,  0.124837041f,  0.0744762495f, 0.0361378156f};
    MYCONST float Scaling            = 1;

    MYCONST bool NeedsScaling = false;
};
template <> struct FixedFilterKernel<mpp::FixedFilter::Gauss, 11, float>
{
    MYCONST float ValuesSeparable[11] = {0.0249794535f, 0.048605375f, 0.0815719739f, 0.118073598f,
                                         0.147407621f,  0.158723891f, 0.147407621f,  0.118073598f,
                                         0.0815719739f, 0.048605375f, 0.0249794535f};
    MYCONST float Scaling             = 1;

    MYCONST bool NeedsScaling = false;
};
template <> struct FixedFilterKernel<mpp::FixedFilter::Gauss, 13, float>
{
    MYCONST float ValuesSeparable[13] = {0.0185440257f, 0.034166947f, 0.0563317686f, 0.0831085443f, 0.109719299f,
                                         0.129618034f,  0.137022823f, 0.129618034f,  0.109719299f,  0.0831085443f,
                                         0.0563317686f, 0.034166947f, 0.0185440257f};
    MYCONST float Scaling             = 1;

    MYCONST bool NeedsScaling = false;
};
template <> struct FixedFilterKernel<mpp::FixedFilter::Gauss, 15, float>
{
    MYCONST float ValuesSeparable[15] = {0.014481457f,  0.0254102573f, 0.0408918671f, 0.0603526309f, 0.0816933662f,
                                         0.101416491f,  0.115468003f,  0.120571882f,  0.115468003f,  0.101416491f,
                                         0.0816933662f, 0.0603526309f, 0.0408918671f, 0.0254102573f, 0.014481457f};
    MYCONST float Scaling             = 1;

    MYCONST bool NeedsScaling = false;
};

template <> struct FixedFilterKernel<mpp::FixedFilter::Gauss, 3, double>
{
    MYCONST double Values[3][3]       = {{0.075113607954112, 0.123841403152974, 0.075113607954112},
                                         {0.123841403152974, 0.204179955571658, 0.123841403152974},
                                         {0.075113607954112, 0.123841403152974, 0.075113607954112}};
    MYCONST double ValuesSeparable[3] = {0.274068619061197, 0.451862761877606, 0.274068619061197};
    MYCONST double Scaling            = 1;

    MYCONST bool NeedsScaling = false;
};
template <> struct FixedFilterKernel<mpp::FixedFilter::Gauss, 5, double>
{
    MYCONST double Values[5][5] = {
        {0.012146124201990, 0.026109944200732, 0.033697319240713, 0.026109944200732, 0.012146124201990},
        {0.026109944200732, 0.056127302407600, 0.072437520846785, 0.056127302407600, 0.026109944200732},
        {0.033697319240713, 0.072437520846785, 0.093487379605793, 0.072437520846785, 0.033697319240713},
        {0.026109944200732, 0.056127302407600, 0.072437520846785, 0.056127302407600, 0.026109944200732},
        {0.012146124201990, 0.026109944200732, 0.033697319240713, 0.026109944200732, 0.012146124201990}};
    MYCONST double ValuesSeparable[5] = {0.110209456046157, 0.236912014063448, 0.305757059780789, 0.236912014063448,
                                         0.110209456046157};
    MYCONST double Scaling            = 1;

    MYCONST bool NeedsScaling = false;
};
template <> struct FixedFilterKernel<mpp::FixedFilter::Gauss, 7, double>
{
    MYCONST double ValuesSeparable[7] = {0.0581035630768955, 0.125691680352928, 0.199695736333686, 0.233018040472982,
                                         0.199695736333686,  0.125691680352928, 0.0581035630768955};
    MYCONST double Scaling            = 1;

    MYCONST bool NeedsScaling = false;
};
template <> struct FixedFilterKernel<mpp::FixedFilter::Gauss, 9, double>
{
    MYCONST double ValuesSeparable[9] = {0.0361378161145520, 0.0744762550723334, 0.124837048684735,
                                         0.170191901323286,  0.188713957610189,  0.170191901323286,
                                         0.124837048684735,  0.0744762550723334, 0.0361378161145520};
    MYCONST double Scaling            = 1;

    MYCONST bool NeedsScaling = false;
};
template <> struct FixedFilterKernel<mpp::FixedFilter::Gauss, 11, double>
{
    MYCONST double ValuesSeparable[11] = {0.0249794576768517, 0.0486053820984010, 0.0815719739785536, 0.118073608642535,
                                          0.147407627175436,  0.158723900856447,  0.147407627175436,  0.118073608642535,
                                          0.0815719739785536, 0.0486053820984010, 0.0249794576768517};
    MYCONST double Scaling             = 1;

    MYCONST bool NeedsScaling = false;
};
template <> struct FixedFilterKernel<mpp::FixedFilter::Gauss, 13, double>
{
    MYCONST double ValuesSeparable[13] = {
        0.0185440216787710, 0.0341669419432277, 0.0563317639365759, 0.0831085392778604, 0.109719294211925,
        0.129618030709527,  0.137022816484225,  0.129618030709527,  0.109719294211925,  0.0831085392778604,
        0.0563317639365759, 0.0341669419432277, 0.0185440216787710};
    MYCONST double Scaling = 1;

    MYCONST bool NeedsScaling = false;
};
template <> struct FixedFilterKernel<mpp::FixedFilter::Gauss, 15, double>
{
    MYCONST double ValuesSeparable[15] = {
        0.0144814557452038, 0.0254102504545484, 0.0408918673316948, 0.0603526318840078, 0.0816933597883408,
        0.101416489755733,  0.115468005654524,  0.120571878771895,  0.115468005654524,  0.101416489755733,
        0.0816933597883408, 0.0603526318840078, 0.0408918673316948, 0.0254102504545484, 0.0144814557452038};
    MYCONST double Scaling = 1;

    MYCONST bool NeedsScaling = false;
};

template <typename filterT> struct FixedFilterKernel<mpp::FixedFilter::HighPass, 3, filterT>
{
    MYCONST filterT Values[3][3] = {{-1, -1, -1}, {-1, 8, -1}, {-1, -1, -1}};
    MYCONST filterT Scaling      = 1;

    MYCONST bool NeedsScaling = false;
};

template <> struct FixedFilterKernel<mpp::FixedFilter::LowPass, 3, int>
{
    MYCONST int Values[3][3] = {{1, 1, 1}, {1, 1, 1}, {1, 1, 1}};
    MYCONST int Scaling      = 9;

    MYCONST bool NeedsScaling = true;
};
template <> struct FixedFilterKernel<mpp::FixedFilter::LowPass, 3, float>
{
    MYCONST float Values[3][3] = {
        {1 / 9.0f, 1 / 9.0f, 1 / 9.0f}, {1 / 9.0f, 1 / 9.0f, 1 / 9.0f}, {1 / 9.0f, 1 / 9.0f, 1 / 9.0f}};
    MYCONST float ValuesSeparable[3] = {1 / 3.0f, 1 / 3.0f, 1 / 3.0f};
    MYCONST float Scaling            = 1;

    MYCONST bool NeedsScaling = false;
};
template <> struct FixedFilterKernel<mpp::FixedFilter::LowPass, 3, double>
{
    MYCONST double Values[3][3] = {
        {1 / 9.0, 1 / 9.0, 1 / 9.0}, {1 / 9.0, 1 / 9.0, 1 / 9.0}, {1 / 9.0, 1 / 9.0, 1 / 9.0}};
    MYCONST double ValuesSeparable[3] = {1 / 3.0, 1 / 3.0, 1 / 3.0};
    MYCONST float Scaling             = 1;

    MYCONST bool NeedsScaling = false;
};

template <typename filterT> struct FixedFilterKernel<mpp::FixedFilter::Laplace, 3, filterT>
{
    MYCONST filterT Values[3][3] = {{-1, -1, -1}, {-1, 8, -1}, {-1, -1, -1}};
    MYCONST filterT Scaling      = 1;

    MYCONST bool NeedsScaling = false;
};

template <typename filterT> struct FixedFilterKernel<mpp::FixedFilter::PrewittHoriz, 3, filterT>
{
    MYCONST filterT Values[3][3] = {{-1, -1, -1}, {0, 0, 0}, {1, 1, 1}};
    MYCONST filterT Scaling      = 1;

    MYCONST bool NeedsScaling = false;
};

template <typename filterT> struct FixedFilterKernel<mpp::FixedFilter::PrewittVert, 3, filterT>
{
    MYCONST filterT Values[3][3] = {{-1, 0, 1}, {-1, 0, 1}, {-1, 0, 1}};
    MYCONST filterT Scaling      = 1;

    MYCONST bool NeedsScaling = false;
};

template <typename filterT> struct FixedInvertedFilterKernel<mpp::FixedFilter::PrewittVert, 3, filterT>
{
    MYCONST filterT Values[3][3] = {{1, 0, -1}, {1, 0, -1}, {1, 0, -1}};
    MYCONST filterT Scaling      = 1;

    MYCONST bool NeedsScaling = false;
};

template <typename filterT> struct FixedFilterKernel<mpp::FixedFilter::RobertsDown, 3, filterT>
{
    MYCONST filterT Values[3][3] = {{-1, 0, 0}, {0, 1, 0}, {0, 0, 0}};
    MYCONST filterT Scaling      = 1;

    MYCONST bool NeedsScaling = false;
};

template <typename filterT> struct FixedFilterKernel<mpp::FixedFilter::RobertsUp, 3, filterT>
{
    MYCONST filterT Values[3][3] = {{0, 0, -1}, {0, 1, 0}, {0, 0, 0}};
    MYCONST filterT Scaling      = 1;

    MYCONST bool NeedsScaling = false;
};

template <typename filterT> struct FixedFilterKernel<mpp::FixedFilter::ScharrHoriz, 3, filterT>
{
    MYCONST filterT Values[3][3] = {{-3, -10, -3}, {0, 0, 0}, {3, 10, 3}};
    MYCONST filterT Scaling      = 1;

    MYCONST bool NeedsScaling = false;
};

template <typename filterT> struct FixedFilterKernel<mpp::FixedFilter::ScharrVert, 3, filterT>
{
    MYCONST filterT Values[3][3] = {{-3, 0, 3}, {-10, 0, 10}, {-3, 0, 3}};
    MYCONST filterT Scaling      = 1;

    MYCONST bool NeedsScaling = false;
};

template <> struct FixedFilterKernel<mpp::FixedFilter::Sharpen, 3, int>
{
    MYCONST int Values[3][3] = {{-1, -1, -1}, {-1, 16, -1}, {-1, -1, -1}};
    MYCONST int Scaling      = 8;

    MYCONST bool NeedsScaling = true;
};
template <> struct FixedFilterKernel<mpp::FixedFilter::Sharpen, 3, float>
{
    MYCONST float Values[3][3] = {
        {-1 / 8.0f, -1 / 8.0f, -1 / 8.0f}, {-1 / 8.0f, 16 / 8.0f, -1 / 8.0f}, {-1 / 8.0f, -1 / 8.0f, -1 / 8.0f}};
    MYCONST float Scaling = 1;

    MYCONST bool NeedsScaling = false;
};
template <> struct FixedFilterKernel<mpp::FixedFilter::Sharpen, 3, double>
{
    MYCONST double Values[3][3] = {
        {-1 / 8.0, -1 / 8.0, -1 / 8.0}, {-1 / 8.0, 16 / 8.0, -1 / 8.0}, {-1 / 8.0, -1 / 8.0, -1 / 8.0}};
    MYCONST double Scaling = 1;

    MYCONST bool NeedsScaling = false;
};

template <typename filterT> struct FixedFilterKernel<mpp::FixedFilter::SobelCross, 3, filterT>
{
    MYCONST filterT Values[3][3] = {{-1, 0, 1}, {0, 0, 0}, {1, 0, -1}};
    MYCONST filterT Scaling      = 1;

    MYCONST bool NeedsScaling = false;
};

template <typename filterT> struct FixedFilterKernel<mpp::FixedFilter::SobelHoriz, 3, filterT>
{
    MYCONST filterT Values[3][3] = {{-1, -2, -1}, {0, 0, 0}, {1, 2, 1}};
    MYCONST filterT Scaling      = 1;

    MYCONST bool NeedsScaling = false;
};

template <typename filterT> struct FixedFilterKernel<mpp::FixedFilter::SobelVert, 3, filterT>
{
    MYCONST filterT Values[3][3] = {{-1, 0, 1}, {-2, 0, 2}, {-1, 0, 1}};
    MYCONST filterT Scaling      = 1;

    MYCONST bool NeedsScaling = false;
};

template <typename filterT> struct FixedInvertedFilterKernel<mpp::FixedFilter::SobelVert, 3, filterT>
{
    MYCONST filterT Values[3][3] = {{1, 0, -1}, {2, 0, -2}, {1, 0, -1}};
    MYCONST filterT Scaling      = 1;

    MYCONST bool NeedsScaling = false;
};

template <typename filterT> struct FixedFilterKernel<mpp::FixedFilter::SobelHorizSecond, 3, filterT>
{
    MYCONST filterT Values[3][3] = {{1, 2, 1}, {-2, -4, -2}, {1, 2, 1}};
    MYCONST filterT Scaling      = 1;

    MYCONST bool NeedsScaling = false;
};

template <typename filterT> struct FixedFilterKernel<mpp::FixedFilter::SobelVertSecond, 3, filterT>
{
    MYCONST filterT Values[3][3] = {{1, -2, 1}, {2, -4, 2}, {1, -2, 1}};
    MYCONST filterT Scaling      = 1;

    MYCONST bool NeedsScaling = false;
};

template <typename filterT> struct FixedFilterKernel<mpp::FixedFilter::HighPass, 5, filterT>
{
    MYCONST filterT Values[5][5] = {{-1, -1, -1, -1, -1}, //
                                    {-1, -1, -1, -1, -1}, //
                                    {-1, -1, 24, -1, -1}, //
                                    {-1, -1, -1, -1, -1}, //
                                    {-1, -1, -1, -1, -1}};
    MYCONST filterT Scaling      = 1;

    MYCONST bool NeedsScaling = false;
};

template <> struct FixedFilterKernel<mpp::FixedFilter::LowPass, 5, int>
{
    MYCONST int Values[5][5]       = {{1, 1, 1, 1, 1}, //
                                      {1, 1, 1, 1, 1}, //
                                      {1, 1, 1, 1, 1}, //
                                      {1, 1, 1, 1, 1}, //
                                      {1, 1, 1, 1, 1}};
    MYCONST int ValuesSeparable[5] = {1, 1, 1, 1, 1};
    MYCONST int Scaling            = 25;

    MYCONST bool NeedsScaling = true;
};
template <> struct FixedFilterKernel<mpp::FixedFilter::LowPass, 5, float>
{
    MYCONST float Values[5][5]       = {{1 / 25.0f, 1 / 25.0f, 1 / 25.0f, 1 / 25.0f, 1 / 25.0f}, //
                                        {1 / 25.0f, 1 / 25.0f, 1 / 25.0f, 1 / 25.0f, 1 / 25.0f}, //
                                        {1 / 25.0f, 1 / 25.0f, 1 / 25.0f, 1 / 25.0f, 1 / 25.0f}, //
                                        {1 / 25.0f, 1 / 25.0f, 1 / 25.0f, 1 / 25.0f, 1 / 25.0f}, //
                                        {1 / 25.0f, 1 / 25.0f, 1 / 25.0f, 1 / 25.0f, 1 / 25.0f}};
    MYCONST float ValuesSeparable[5] = {1 / 5.0f, 1 / 5.0f, 1 / 5.0f, 1 / 5.0f, 1 / 5.0f};
    MYCONST float Scaling            = 1;

    MYCONST bool NeedsScaling = false;
};
template <> struct FixedFilterKernel<mpp::FixedFilter::LowPass, 5, double>
{
    MYCONST double Values[5][5]       = {{1 / 25.0, 1 / 25.0, 1 / 25.0, 1 / 25.0, 1 / 25.0}, //
                                         {1 / 25.0, 1 / 25.0, 1 / 25.0, 1 / 25.0, 1 / 25.0}, //
                                         {1 / 25.0, 1 / 25.0, 1 / 25.0, 1 / 25.0, 1 / 25.0}, //
                                         {1 / 25.0, 1 / 25.0, 1 / 25.0, 1 / 25.0, 1 / 25.0}, //
                                         {1 / 25.0, 1 / 25.0, 1 / 25.0, 1 / 25.0, 1 / 25.0}};
    MYCONST double ValuesSeparable[5] = {1 / 5.0, 1 / 5.0, 1 / 5.0, 1 / 5.0, 1 / 5.0};
    MYCONST double Scaling            = 1;

    MYCONST bool NeedsScaling = false;
};

template <typename filterT> struct FixedFilterKernel<mpp::FixedFilter::Laplace, 5, filterT>
{
    MYCONST filterT Values[5][5] = {{-1, -3, -4, -3, -1}, //
                                    {-3, 0, 6, 0, -3},    //
                                    {-4, 6, 20, 6, -4},   //
                                    {-3, 0, 6, 0, -3},    //
                                    {-1, -3, -4, -3, -1}};
    MYCONST filterT Scaling      = 1;

    MYCONST bool NeedsScaling = false;
};

template <typename filterT> struct FixedFilterKernel<mpp::FixedFilter::SobelCross, 5, filterT>
{
    MYCONST filterT Values[5][5] = {{-1, -2, 0, 2, 1}, //
                                    {-2, -4, 0, 4, 2}, //
                                    {0, 0, 0, 0, 0},   //
                                    {2, 4, 0, -4, -2}, //
                                    {1, 2, 0, -2, -1}};
    MYCONST filterT Scaling      = 1;

    MYCONST bool NeedsScaling = false;
};

template <typename filterT> struct FixedFilterKernel<mpp::FixedFilter::SobelHoriz, 5, filterT>
{
    MYCONST filterT Values[5][5] = {{-1, -4, -6, -4, -1},  //
                                    {-2, -8, -12, -8, -2}, //
                                    {0, 0, 0, 0, 0},       //
                                    {2, 8, 12, 8, 2},      //
                                    {1, 4, 6, 4, 1}};
    MYCONST filterT Scaling      = 1;

    MYCONST bool NeedsScaling = false;
};

template <typename filterT> struct FixedFilterKernel<mpp::FixedFilter::SobelVert, 5, filterT>
{
    MYCONST filterT Values[5][5] = {{-1, -2, 0, 2, 1},   //
                                    {-4, -8, 0, 8, 4},   //
                                    {-6, -12, 0, 12, 6}, //
                                    {-4, -8, 0, 8, 4},   //
                                    {-1, -2, 0, 2, 1}};
    MYCONST filterT Scaling      = 1;

    MYCONST bool NeedsScaling = false;
};

template <typename filterT> struct FixedInvertedFilterKernel<mpp::FixedFilter::SobelVert, 5, filterT>
{
    MYCONST filterT Values[5][5] = {{1, 2, 0, -2, -1},   //
                                    {4, 8, 0, -8, -4},   //
                                    {6, 12, 0, -12, -6}, //
                                    {4, 8, 0, -8, -4},   //
                                    {1, 2, 0, -2, -1}};
    MYCONST filterT Scaling      = 1;

    MYCONST bool NeedsScaling = false;
};

template <typename filterT> struct FixedFilterKernel<mpp::FixedFilter::SobelHorizSecond, 5, filterT>
{
    MYCONST filterT Values[5][5] = {{1, 4, 6, 4, 1},       //
                                    {0, 0, 0, 0, 0},       //
                                    {-2, -8, -12, -8, -2}, //
                                    {0, 0, 0, 0, 0},       //
                                    {1, 4, 6, 4, 1}};
    MYCONST filterT Scaling      = 1;

    MYCONST bool NeedsScaling = false;
};

template <typename filterT> struct FixedFilterKernel<mpp::FixedFilter::SobelVertSecond, 5, filterT>
{
    MYCONST filterT Values[5][5] = {{1, 0, -2, 0, 1},  //
                                    {4, 0, -8, 0, 4},  //
                                    {6, 0, -12, 0, 6}, //
                                    {4, 0, -8, 0, 4},  //
                                    {1, 0, -2, 0, 1}};
    MYCONST filterT Scaling      = 1;

    MYCONST bool NeedsScaling = false;
};

template <> struct FixedFilterKernel<mpp::FixedFilter::LowPass, 7, float>
{
    MYCONST float ValuesSeparable[7] = {1 / 7.0f, 1 / 7.0f, 1 / 7.0f, 1 / 7.0f, 1 / 7.0f, 1 / 7.0f, 1 / 7.0f};

    MYCONST bool NeedsScaling = false;
};

template <> struct FixedFilterKernel<mpp::FixedFilter::LowPass, 9, float>
{
    MYCONST float ValuesSeparable[9] = {1 / 9.0f, 1 / 9.0f, 1 / 9.0f, 1 / 9.0f, 1 / 9.0f,
                                        1 / 9.0f, 1 / 9.0f, 1 / 9.0f, 1 / 9.0f};

    MYCONST bool NeedsScaling = false;
};

template <> struct FixedFilterKernel<mpp::FixedFilter::LowPass, 11, float>
{
    MYCONST float ValuesSeparable[11] = {1 / 11.0f, 1 / 11.0f, 1 / 11.0f, 1 / 11.0f, 1 / 11.0f, 1 / 11.0f,
                                         1 / 11.0f, 1 / 11.0f, 1 / 11.0f, 1 / 11.0f, 1 / 11.0f};

    MYCONST bool NeedsScaling = false;
};

template <> struct FixedFilterKernel<mpp::FixedFilter::LowPass, 13, float>
{
    MYCONST float ValuesSeparable[13] = {1 / 13.0f, 1 / 13.0f, 1 / 13.0f, 1 / 13.0f, 1 / 13.0f, 1 / 13.0f, 1 / 13.0f,
                                         1 / 13.0f, 1 / 13.0f, 1 / 13.0f, 1 / 13.0f, 1 / 13.0f, 1 / 13.0f};

    MYCONST bool NeedsScaling = false;
};

template <> struct FixedFilterKernel<mpp::FixedFilter::LowPass, 15, float>
{
    MYCONST float ValuesSeparable[15] = {1 / 15.0f, 1 / 15.0f, 1 / 15.0f, 1 / 15.0f, 1 / 15.0f,
                                         1 / 15.0f, 1 / 15.0f, 1 / 15.0f, 1 / 15.0f, 1 / 15.0f,
                                         1 / 15.0f, 1 / 15.0f, 1 / 15.0f, 1 / 15.0f, 1 / 15.0f};

    MYCONST bool NeedsScaling = false;
};

template <> struct FixedFilterKernel<mpp::FixedFilter::LowPass, 7, double>
{
    MYCONST double ValuesSeparable[7] = {1 / 7.0, 1 / 7.0, 1 / 7.0, 1 / 7.0, 1 / 7.0, 1 / 7.0, 1 / 7.0};

    MYCONST bool NeedsScaling = false;
};

template <> struct FixedFilterKernel<mpp::FixedFilter::LowPass, 9, double>
{
    MYCONST double ValuesSeparable[9] = {1 / 9.0, 1 / 9.0, 1 / 9.0, 1 / 9.0, 1 / 9.0,
                                         1 / 9.0, 1 / 9.0, 1 / 9.0, 1 / 9.0};

    MYCONST bool NeedsScaling = false;
};

template <> struct FixedFilterKernel<mpp::FixedFilter::LowPass, 11, double>
{
    MYCONST double ValuesSeparable[11] = {1 / 11.0, 1 / 11.0, 1 / 11.0, 1 / 11.0, 1 / 11.0, 1 / 11.0,
                                          1 / 11.0, 1 / 11.0, 1 / 11.0, 1 / 11.0, 1 / 11.0};

    MYCONST bool NeedsScaling = false;
};

template <> struct FixedFilterKernel<mpp::FixedFilter::LowPass, 13, double>
{
    MYCONST double ValuesSeparable[13] = {1 / 13.0, 1 / 13.0, 1 / 13.0, 1 / 13.0, 1 / 13.0, 1 / 13.0, 1 / 13.0,
                                          1 / 13.0, 1 / 13.0, 1 / 13.0, 1 / 13.0, 1 / 13.0, 1 / 13.0};

    MYCONST bool NeedsScaling = false;
};

template <> struct FixedFilterKernel<mpp::FixedFilter::LowPass, 15, double>
{
    MYCONST double ValuesSeparable[15] = {1 / 15.0, 1 / 15.0, 1 / 15.0, 1 / 15.0, 1 / 15.0,
                                          1 / 15.0, 1 / 15.0, 1 / 15.0, 1 / 15.0, 1 / 15.0,
                                          1 / 15.0, 1 / 15.0, 1 / 15.0, 1 / 15.0, 1 / 15.0};

    MYCONST bool NeedsScaling = false;
};

#undef MYCONST
} // namespace mpp::image