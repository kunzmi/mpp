#pragma once
#include <common/defines.h>
#include <ostream>

namespace mpp
{
#pragma region RoundingMode
/// <summary>
/// Rounding Modes<para/>
/// The enumerated rounding modes are used by a large number of MPP primitives
/// to allow the user to specify the method by which fractional values are converted
/// to integer values.
/// </summary>
enum class RoundingMode // NOLINT(performance-enum-size)
{
    /// <summary>
    /// Round to the nearest even integer.<para/>
    /// All fractional numbers are rounded to their nearest integer. The ambiguous
    /// cases (i.e. integer.5) are rounded to the closest even integer.<para/>
    /// __float2int_rn in CUDA<para/>
    /// E.g.<para/>
    /// - roundNear(0.4) = 0<para/>
    /// - roundNear(0.5) = 0<para/>
    /// - roundNear(0.6) = 1<para/>
    /// - roundNear(1.5) = 2<para/>
    /// - roundNear(1.9) = 2<para/>
    /// - roundNear(-1.5) = -2<para/>
    /// - roundNear(-2.5) = -2<para/>
    /// </summary>
    NearestTiesToEven,
    /// <summary>
    /// Round according to financial rule.<para/>
    /// All fractional numbers are rounded to their nearest integer. The ambiguous
    /// cases (i.e. integer.5) are rounded away from zero.<para/>
    /// C++ round() function<para/>
    /// E.g.<para/>
    /// - roundNearestTiesAwayFromZero(0.4)  = 0<para/>
    /// - roundNearestTiesAwayFromZero(0.5)  = 1<para/>
    /// - roundNearestTiesAwayFromZero(0.6)  = 1<para/>
    /// - roundNearestTiesAwayFromZero(1.5)  = 2<para/>
    /// - roundNearestTiesAwayFromZero(1.9)  = 2<para/>
    /// - roundNearestTiesAwayFromZero(-1.5) = -2<para/>
    /// - roundNearestTiesAwayFromZero(-2.5) = -3<para/>
    /// </summary>
    NearestTiesAwayFromZero,
    /// <summary>
    /// Round towards zero (truncation).<para/>
    /// All fractional numbers of the form integer.decimals are truncated to
    /// integer.<para/>
    /// __float2int_rz in CUDA<para/>
    /// - roundZero(0.4)  = 0<para/>
    /// - roundZero(0.5)  = 0<para/>
    /// - roundZero(0.6)  = 0<para/>
    /// - roundZero(1.5)  = 1<para/>
    /// - roundZero(1.9)  = 1<para/>
    /// - roundZero(-1.5) = -1<para/>
    /// - roundZero(-2.5) = -2<para/>
    /// </summary>
    TowardZero,
    /// <summary>
    /// Round towards negative infinity.<para/>
    /// C++ floor() function<para/>
    /// E.g.<para/>
    /// - floor(0.4)  = 0<para/>
    /// - floor(0.5)  = 0<para/>
    /// - floor(0.6)  = 0<para/>
    /// - floor(1.5)  = 1<para/>
    /// - floor(1.9)  = 1<para/>
    /// - floor(-1.5) = -2<para/>
    /// - floor(-2.5) = -3<para/>
    /// </summary>
    TowardNegativeInfinity,
    /// <summary>
    /// Round towards positive infinity.<para/>
    /// C++ ceil() function<para/>
    /// E.g.<para/>
    /// - ceil(0.4)  = 1<para/>
    /// - ceil(0.5)  = 1<para/>
    /// - ceil(0.6)  = 1<para/>
    /// - ceil(1.5)  = 2<para/>
    /// - ceil(1.9)  = 2<para/>
    /// - ceil(-1.5) = -1<para/>
    /// - ceil(-2.5) = -2<para/>
    /// </summary>
    TowardPositiveInfinity,
    /// <summary>
    /// No rounding at all, NOP, for internal use
    /// </summary>
    None
};

template <RoundingMode T> struct rouding_mode_name
{
    static constexpr char value[] = "Unknown";
};
template <> struct rouding_mode_name<RoundingMode::NearestTiesToEven>
{
    static constexpr char value[] = "NearestTiesToEven";
};
template <> struct rouding_mode_name<RoundingMode::NearestTiesAwayFromZero>
{
    static constexpr char value[] = "NearestTiesAwayFromZero";
};
template <> struct rouding_mode_name<RoundingMode::TowardZero>
{
    static constexpr char value[] = "TowardZero";
};
template <> struct rouding_mode_name<RoundingMode::TowardNegativeInfinity>
{
    static constexpr char value[] = "TowardNegativeInfinity";
};
template <> struct rouding_mode_name<RoundingMode::TowardPositiveInfinity>
{
    static constexpr char value[] = "TowardPositiveInfinity";
};
template <> struct rouding_mode_name<RoundingMode::None>
{
    static constexpr char value[] = "None";
};

inline std::ostream &operator<<(std::ostream &aOs, const RoundingMode &aRoundingMode)
{
    switch (aRoundingMode)
    {
        case RoundingMode::NearestTiesToEven:
            aOs << rouding_mode_name<RoundingMode::NearestTiesToEven>::value;
            break;
        case RoundingMode::NearestTiesAwayFromZero:
            aOs << rouding_mode_name<RoundingMode::NearestTiesAwayFromZero>::value;
            break;
        case RoundingMode::TowardZero:
            aOs << rouding_mode_name<RoundingMode::TowardZero>::value;
            break;
        case RoundingMode::TowardNegativeInfinity:
            aOs << rouding_mode_name<RoundingMode::TowardNegativeInfinity>::value;
            break;
        case RoundingMode::TowardPositiveInfinity:
            aOs << rouding_mode_name<RoundingMode::TowardPositiveInfinity>::value;
            break;
        case RoundingMode::None:
            aOs << rouding_mode_name<RoundingMode::None>::value;
            break;
        default:
        {
            const std::ios::fmtflags f(aOs.flags());
            aOs << "Unknown: 0x" << std::hex << std::uppercase << static_cast<uint>(aRoundingMode);
            aOs.flags(f);
        }
        break;
    }
    return aOs;
}

inline std::wostream &operator<<(std::wostream &aOs, const RoundingMode &aRoundingMode)
{
    switch (aRoundingMode)
    {
        case RoundingMode::NearestTiesToEven:
            aOs << rouding_mode_name<RoundingMode::NearestTiesToEven>::value;
            break;
        case RoundingMode::NearestTiesAwayFromZero:
            aOs << rouding_mode_name<RoundingMode::NearestTiesAwayFromZero>::value;
            break;
        case RoundingMode::TowardZero:
            aOs << rouding_mode_name<RoundingMode::TowardZero>::value;
            break;
        case RoundingMode::TowardNegativeInfinity:
            aOs << rouding_mode_name<RoundingMode::TowardNegativeInfinity>::value;
            break;
        case RoundingMode::TowardPositiveInfinity:
            aOs << rouding_mode_name<RoundingMode::TowardPositiveInfinity>::value;
            break;
        case RoundingMode::None:
            aOs << rouding_mode_name<RoundingMode::None>::value;
            break;
        default:
        {
            const std::ios::fmtflags f(aOs.flags());
            aOs << "Unknown: 0x" << std::hex << std::uppercase << static_cast<uint>(aRoundingMode);
            aOs.flags(f);
        }
        break;
    }
    return aOs;
}
#pragma endregion

#pragma region AlphaComposition

/// <summary>
/// Different Alpha compositing operations
/// </summary>
enum class AlphaCompositionOp // NOLINT(performance-enum-size)
{
    /// <summary>
    /// OVER compositing operation.<para/>
    /// A occludes B.<para/>
    /// result pixel = alphaA * A + (1 - alphaA) * alphaB * B<para/>
    /// result alpha = alphaA + (1 - alphaA) * alphaB
    /// </summary>
    Over,
    /// <summary>
    /// IN compositing operation.<para/>
    /// A within B. A acts as a matte for B. A shows only where B is visible.<para/>
    /// result pixel = alphaA * A * alphaB<para/>
    /// result alpha = alphaA * alphaB
    /// </summary>
    In,
    /// <summary>
    /// OUT compositing operation.<para/>
    /// A outside B. NOT-B acts as a matte for A. A shows only where B is not visible.<para/>
    /// result pixel = alphaA * A * (1 - alphaB)<para/>
    /// result alpha = alphaA * (1 - alphaB)
    /// </summary>
    Out,
    /// <summary>
    /// ATOP compositing operation.<para/>
    /// Combination of (A IN B) and (B OUT A). B is both back-ground and matte for A.<para/>
    /// result pixel = alphaA * A * alphaB + (1 - alphaA) * alphaB * B<para/>
    /// result alpha = alphaA * alphaB + (1 - alphaA) * alphaB
    /// </summary>
    ATop,
    /// <summary>
    /// XOR compositing operation.<para/>
    /// Combination of (A OUT B) and (B OUT A). A and B mutually exclude each other.<para/>
    /// result pixel = alphaA * A * (1 - alphaB) + (1 - alphaA) * alphaB * B<para/>
    /// result alpha = alphaA * (1 - alphaB) + (1 - alphaA) * alphaB
    /// </summary>
    XOr,
    /// <summary>
    /// PLUS compositing operation.<para/>
    /// Blend without precedence.<para/>
    /// result pixel = alphaA * A + alphaB * B<para/>
    /// result alpha = alphaA + alphaB
    /// </summary>
    Plus,
    /// <summary>
    /// OVER compositing operation with pre-multiplied pixel values.<para/>
    /// result pixel = A + (1 - alphaA) * B<para/>
    /// result alpha = alphaA + (1 - alphaA) * aB
    /// </summary>
    OverPremul,
    /// <summary>
    /// IN compositing operation with pre-multiplied pixel values.<para/>
    /// A within B. A acts as a matte for B. A shows only where B is visible.<para/>
    /// result pixel = A * alphaB<para/>
    /// result alpha = alphaA * alphaB
    /// </summary>
    InPremul,
    /// <summary>
    /// OUT compositing operation with pre-multiplied pixel values.<para/>
    /// A outside B. NOT-B acts as a matte for A. A shows only where B is not visible.<para/>
    /// result pixel = A * (1 - alphaB)<para/>
    /// result alpha = alphaA * (1 - alphaB)
    /// </summary>
    OutPremul,
    /// <summary>
    /// ATOP compositing operation with pre-multiplied pixel values.<para/>
    /// Combination of (A IN B) and (B OUT A). B is both back-ground and matte for A.<para/>
    /// result pixel = A * alphaB + (1 - alphaA) * B<para/>
    /// result alpha = alphaA * alphaB + (1 - alphaA) * alphaB
    /// </summary>
    ATopPremul,
    /// <summary>
    /// XOR compositing operation with pre-multiplied pixel values.<para/>
    /// Combination of (A OUT B) and (B OUT A). A and B mutually exclude each other.<para/>
    /// result pixel = A * (1 - alphaB) + (1 - alphaA) * B<para/>
    /// result alpha = alphaA * (1 - alphaB) + (1 - alphaA) * alphaB
    /// </summary>
    XOrPremul,
    /// <summary>
    /// PLUS compositing operation with pre-multiplied pixel values.<para/>
    /// Blend without precedence.<para/>
    /// result pixel = A + B<para/>
    /// result alpha = alphaA + alphaB
    /// </summary>
    PlusPremul
};

template <AlphaCompositionOp T> struct alpha_composition_name
{
    static constexpr char value[] = "Unknown";
};
template <> struct alpha_composition_name<AlphaCompositionOp::Over>
{
    static constexpr char value[] = "Over";
};
template <> struct alpha_composition_name<AlphaCompositionOp::In>
{
    static constexpr char value[] = "In";
};
template <> struct alpha_composition_name<AlphaCompositionOp::Out>
{
    static constexpr char value[] = "Out";
};
template <> struct alpha_composition_name<AlphaCompositionOp::ATop>
{
    static constexpr char value[] = "Atop";
};
template <> struct alpha_composition_name<AlphaCompositionOp::XOr>
{
    static constexpr char value[] = "Xor";
};
template <> struct alpha_composition_name<AlphaCompositionOp::Plus>
{
    static constexpr char value[] = "Plus";
};
template <> struct alpha_composition_name<AlphaCompositionOp::OverPremul>
{
    static constexpr char value[] = "OverPremul";
};
template <> struct alpha_composition_name<AlphaCompositionOp::InPremul>
{
    static constexpr char value[] = "InPremul";
};
template <> struct alpha_composition_name<AlphaCompositionOp::OutPremul>
{
    static constexpr char value[] = "OutPremul";
};
template <> struct alpha_composition_name<AlphaCompositionOp::ATopPremul>
{
    static constexpr char value[] = "AtopPremul";
};
template <> struct alpha_composition_name<AlphaCompositionOp::XOrPremul>
{
    static constexpr char value[] = "XorPremul";
};
template <> struct alpha_composition_name<AlphaCompositionOp::PlusPremul>
{
    static constexpr char value[] = "PlusPremul";
};

inline std::ostream &operator<<(std::ostream &aOs, const AlphaCompositionOp &aAlphaComposition)
{
    switch (aAlphaComposition)
    {
        case mpp::AlphaCompositionOp::Over:
            aOs << alpha_composition_name<AlphaCompositionOp::Over>::value;
            break;
        case mpp::AlphaCompositionOp::In:
            aOs << alpha_composition_name<AlphaCompositionOp::In>::value;
            break;
        case mpp::AlphaCompositionOp::Out:
            aOs << alpha_composition_name<AlphaCompositionOp::Out>::value;
            break;
        case mpp::AlphaCompositionOp::ATop:
            aOs << alpha_composition_name<AlphaCompositionOp::ATop>::value;
            break;
        case mpp::AlphaCompositionOp::XOr:
            aOs << alpha_composition_name<AlphaCompositionOp::XOr>::value;
            break;
        case mpp::AlphaCompositionOp::Plus:
            aOs << alpha_composition_name<AlphaCompositionOp::Plus>::value;
            break;
        case mpp::AlphaCompositionOp::OverPremul:
            aOs << alpha_composition_name<AlphaCompositionOp::OverPremul>::value;
            break;
        case mpp::AlphaCompositionOp::InPremul:
            aOs << alpha_composition_name<AlphaCompositionOp::InPremul>::value;
            break;
        case mpp::AlphaCompositionOp::OutPremul:
            aOs << alpha_composition_name<AlphaCompositionOp::OutPremul>::value;
            break;
        case mpp::AlphaCompositionOp::ATopPremul:
            aOs << alpha_composition_name<AlphaCompositionOp::ATopPremul>::value;
            break;
        case mpp::AlphaCompositionOp::XOrPremul:
            aOs << alpha_composition_name<AlphaCompositionOp::XOrPremul>::value;
            break;
        case mpp::AlphaCompositionOp::PlusPremul:
            aOs << alpha_composition_name<AlphaCompositionOp::PlusPremul>::value;
            break;
        default:
        {
            const std::ios::fmtflags f(aOs.flags());
            aOs << "Unknown: 0x" << std::hex << std::uppercase << static_cast<uint>(aAlphaComposition);
            aOs.flags(f);
        }
        break;
    }
    return aOs;
}

inline std::wostream &operator<<(std::wostream &aOs, const AlphaCompositionOp &aAlphaComposition)
{
    switch (aAlphaComposition)
    {
        case mpp::AlphaCompositionOp::Over:
            aOs << alpha_composition_name<AlphaCompositionOp::Over>::value;
            break;
        case mpp::AlphaCompositionOp::In:
            aOs << alpha_composition_name<AlphaCompositionOp::In>::value;
            break;
        case mpp::AlphaCompositionOp::Out:
            aOs << alpha_composition_name<AlphaCompositionOp::Out>::value;
            break;
        case mpp::AlphaCompositionOp::ATop:
            aOs << alpha_composition_name<AlphaCompositionOp::ATop>::value;
            break;
        case mpp::AlphaCompositionOp::XOr:
            aOs << alpha_composition_name<AlphaCompositionOp::XOr>::value;
            break;
        case mpp::AlphaCompositionOp::Plus:
            aOs << alpha_composition_name<AlphaCompositionOp::Plus>::value;
            break;
        case mpp::AlphaCompositionOp::OverPremul:
            aOs << alpha_composition_name<AlphaCompositionOp::OverPremul>::value;
            break;
        case mpp::AlphaCompositionOp::InPremul:
            aOs << alpha_composition_name<AlphaCompositionOp::InPremul>::value;
            break;
        case mpp::AlphaCompositionOp::OutPremul:
            aOs << alpha_composition_name<AlphaCompositionOp::OutPremul>::value;
            break;
        case mpp::AlphaCompositionOp::ATopPremul:
            aOs << alpha_composition_name<AlphaCompositionOp::ATopPremul>::value;
            break;
        case mpp::AlphaCompositionOp::XOrPremul:
            aOs << alpha_composition_name<AlphaCompositionOp::XOrPremul>::value;
            break;
        case mpp::AlphaCompositionOp::PlusPremul:
            aOs << alpha_composition_name<AlphaCompositionOp::PlusPremul>::value;
            break;
        default:
        {
            const std::ios::fmtflags f(aOs.flags());
            aOs << "Unknown: 0x" << std::hex << std::uppercase << static_cast<uint>(aAlphaComposition);
            aOs.flags(f);
        }
        break;
    }
    return aOs;
}
#pragma endregion

#pragma region BayerGridPosition
/// <summary>
/// Bayer grid position registration
/// </summary>
enum class BayerGridPosition // NOLINT(performance-enum-size)
{
    /// <summary>
    /// Blue Green<para/>
    /// Green Red<para/>
    /// Bayer pattern.
    /// </summary>
    BGGR,
    /// <summary>
    /// Red Green<para/>
    /// Green Blue<para/>
    /// Bayer pattern.
    /// </summary>
    RGGB,
    /// <summary>
    /// Green Blue<para/>
    /// Red Green<para/>
    /// Bayer pattern.
    /// </summary>
    GBRG,
    /// <summary>
    /// Green Red<para/>
    /// Blue Green<para/>
    /// Bayer pattern.
    /// </summary>
    GRBG
};

template <BayerGridPosition T> struct bayer_grid_position_name
{
    static constexpr char value[] = "Unknown";
};
template <> struct bayer_grid_position_name<BayerGridPosition::BGGR>
{
    static constexpr char value[] = "BGGR";
};
template <> struct bayer_grid_position_name<BayerGridPosition::RGGB>
{
    static constexpr char value[] = "RGGB";
};
template <> struct bayer_grid_position_name<BayerGridPosition::GBRG>
{
    static constexpr char value[] = "GBRG";
};
template <> struct bayer_grid_position_name<BayerGridPosition::GRBG>
{
    static constexpr char value[] = "GRBG";
};

inline std::ostream &operator<<(std::ostream &aOs, const BayerGridPosition &aBayerGridPosition)
{
    switch (aBayerGridPosition)
    {
        case mpp::BayerGridPosition::BGGR:
            aOs << bayer_grid_position_name<BayerGridPosition::BGGR>::value;
            break;
        case mpp::BayerGridPosition::RGGB:
            aOs << bayer_grid_position_name<BayerGridPosition::RGGB>::value;
            break;
        case mpp::BayerGridPosition::GBRG:
            aOs << bayer_grid_position_name<BayerGridPosition::GBRG>::value;
            break;
        case mpp::BayerGridPosition::GRBG:
            aOs << bayer_grid_position_name<BayerGridPosition::GRBG>::value;
            break;
        default:
        {
            const std::ios::fmtflags f(aOs.flags());
            aOs << "Unknown: 0x" << std::hex << std::uppercase << static_cast<uint>(aBayerGridPosition);
            aOs.flags(f);
        }
        break;
    }
    return aOs;
}

inline std::wostream &operator<<(std::wostream &aOs, const BayerGridPosition &aBayerGridPosition)
{
    switch (aBayerGridPosition)
    {
        case mpp::BayerGridPosition::BGGR:
            aOs << bayer_grid_position_name<BayerGridPosition::BGGR>::value;
            break;
        case mpp::BayerGridPosition::RGGB:
            aOs << bayer_grid_position_name<BayerGridPosition::RGGB>::value;
            break;
        case mpp::BayerGridPosition::GBRG:
            aOs << bayer_grid_position_name<BayerGridPosition::GBRG>::value;
            break;
        case mpp::BayerGridPosition::GRBG:
            aOs << bayer_grid_position_name<BayerGridPosition::GRBG>::value;
            break;
        default:
        {
            const std::ios::fmtflags f(aOs.flags());
            aOs << "Unknown: 0x" << std::hex << std::uppercase << static_cast<uint>(aBayerGridPosition);
            aOs.flags(f);
        }
        break;
    }
    return aOs;
}
#pragma endregion

#pragma region MirrorAxis
/// <summary>
/// Mirror direction control
/// </summary>
enum class MirrorAxis // NOLINT(performance-enum-size)
{
    /// <summary>
    /// Flip around horizontal axis in mirror function.
    /// </summary>
    Horizontal,
    /// <summary>
    /// Flip around vertical axis in mirror function.
    /// </summary>
    Vertical,
    /// <summary>
    /// Flip around both axes in mirror function.
    /// </summary>
    Both
};

template <MirrorAxis T> struct mirror_axis_name
{
    static constexpr char value[] = "Unknown";
};
template <> struct mirror_axis_name<MirrorAxis::Horizontal>
{
    static constexpr char value[] = "Horizontal";
};
template <> struct mirror_axis_name<MirrorAxis::Vertical>
{
    static constexpr char value[] = "Vertical";
};
template <> struct mirror_axis_name<MirrorAxis::Both>
{
    static constexpr char value[] = "Both";
};

inline std::ostream &operator<<(std::ostream &aOs, const MirrorAxis &aMirrorAxis)
{
    switch (aMirrorAxis)
    {
        case mpp::MirrorAxis::Horizontal:
            aOs << mirror_axis_name<MirrorAxis::Horizontal>::value;
            break;
        case mpp::MirrorAxis::Vertical:
            aOs << mirror_axis_name<MirrorAxis::Vertical>::value;
            break;
        case mpp::MirrorAxis::Both:
            aOs << mirror_axis_name<MirrorAxis::Both>::value;
            break;
        default:
        {
            const std::ios::fmtflags f(aOs.flags());
            aOs << "Unknown: 0x" << std::hex << std::uppercase << static_cast<uint>(aMirrorAxis);
            aOs.flags(f);
        }
        break;
    }
    return aOs;
}

inline std::wostream &operator<<(std::wostream &aOs, const MirrorAxis &aMirrorAxis)
{
    switch (aMirrorAxis)
    {
        case mpp::MirrorAxis::Horizontal:
            aOs << mirror_axis_name<MirrorAxis::Horizontal>::value;
            break;
        case mpp::MirrorAxis::Vertical:
            aOs << mirror_axis_name<MirrorAxis::Vertical>::value;
            break;
        case mpp::MirrorAxis::Both:
            aOs << mirror_axis_name<MirrorAxis::Both>::value;
            break;
        default:
        {
            const std::ios::fmtflags f(aOs.flags());
            aOs << "Unknown: 0x" << std::hex << std::uppercase << static_cast<uint>(aMirrorAxis);
            aOs.flags(f);
        }
        break;
    }
    return aOs;
}
#pragma endregion

#pragma region CompareOp
/// <summary>
/// Pixel comparison control values
/// </summary>
enum class CompareOp : uint // NOLINT(performance-enum-size)
{
    /// <summary>
    /// Returns true if the pixel value is &lt; than the value to compare with.
    /// </summary>
    Less = 1u,
    /// <summary>
    /// Returns true if the pixel value is &lt;= than the value to compare with.
    /// </summary>
    LessEq = 1u << 1u,
    /// <summary>
    /// Returns true if the pixel value is == than the value to compare with.
    /// </summary>
    Eq = 1u << 2u,
    /// <summary>
    /// Returns true if the pixel value is &gt; than the value to compare with.
    /// </summary>
    Greater = 1u << 3u,
    /// <summary>
    /// Returns true if the pixel value is &gt;= than the value to compare with.
    /// </summary>
    GreaterEq = 1u << 4u,
    /// <summary>
    /// Returns true if the pixel value is != than the value to compare with.
    /// </summary>
    NEq = 1u << 5u,
    /// <summary>
    /// Returns true if the pixel value is finite (only for floating point).
    /// </summary>
    IsFinite = 1u << 6u,
    /// <summary>
    /// Returns true if the pixel value is NaN (only for floating point).
    /// </summary>
    IsNaN = 1u << 7u,
    /// <summary>
    /// Returns true if the pixel value is infinite (only for floating point).
    /// </summary>
    IsInf = 1u << 8u,
    /// <summary>
    /// Returns true if the pixel value is infinite or NaN (i.e. not finite) (only for floating point).
    /// </summary>
    IsInfOrNaN = 1u << 9u,
    /// <summary>
    /// Returns true if the pixel value is positive infinite (only for floating point).
    /// </summary>
    IsPositiveInf = 1u << 10u,
    /// <summary>
    /// Returns true if the pixel value is negative infinite (only for floating point).
    /// </summary>
    IsNegativeInf = 1u << 11u,
    /// <summary>
    /// If PerChannel flag is set, the comparison is performed per channel independently.
    /// </summary>
    PerChannel = 1u << 26u,
    /// <summary>
    /// If AnyChannel flag is set, the comparison returns true if any of the pixel channel comparisons is true. If not
    /// set, all pixel channel comparisons must be true.
    /// </summary>
    AnyChannel = 1u << 27u
};

constexpr CompareOp operator|(CompareOp aLeft, CompareOp aRight) noexcept
{
    return static_cast<CompareOp>(static_cast<uint>(aLeft) | static_cast<uint>(aRight));
}

constexpr CompareOp operator&(CompareOp aLeft, CompareOp aRight) noexcept
{
    return static_cast<CompareOp>(static_cast<uint>(aLeft) & static_cast<uint>(aRight));
}

constexpr bool CompareOp_IsPerChannel(CompareOp aCompareOp)
{
    return (static_cast<uint>(aCompareOp) & static_cast<uint>(CompareOp::PerChannel)) != 0;
}

constexpr bool CompareOp_IsAnyChannel(CompareOp aCompareOp)
{
    return (static_cast<uint>(aCompareOp) & static_cast<uint>(CompareOp::AnyChannel)) != 0;
}

constexpr CompareOp CompareOp_NoFlags(CompareOp aCompareOp)
{
    // 0xFFFF is smaller than the first flag and includes all bits of non-flag values.
    return static_cast<CompareOp>(static_cast<uint>(aCompareOp) & 0xFFFFu);
}

template <CompareOp T> struct compare_op_name
{
    static constexpr char value[] = "Unknown";
};
template <> struct compare_op_name<CompareOp::Less>
{
    static constexpr char value[] = "Less";
};
template <> struct compare_op_name<CompareOp::LessEq>
{
    static constexpr char value[] = "LessEq";
};
template <> struct compare_op_name<CompareOp::Eq>
{
    static constexpr char value[] = "Eq";
};
template <> struct compare_op_name<CompareOp::Greater>
{
    static constexpr char value[] = "Greater";
};
template <> struct compare_op_name<CompareOp::GreaterEq>
{
    static constexpr char value[] = "GreaterEq";
};
template <> struct compare_op_name<CompareOp::NEq>
{
    static constexpr char value[] = "NEq";
};
template <> struct compare_op_name<CompareOp::IsFinite>
{
    static constexpr char value[] = "IsFinite";
};
template <> struct compare_op_name<CompareOp::IsNaN>
{
    static constexpr char value[] = "IsNaN";
};
template <> struct compare_op_name<CompareOp::IsInf>
{
    static constexpr char value[] = "IsInf";
};
template <> struct compare_op_name<CompareOp::IsInfOrNaN>
{
    static constexpr char value[] = "IsInfOrNaN";
};
template <> struct compare_op_name<CompareOp::IsPositiveInf>
{
    static constexpr char value[] = "IsPositiveInf";
};
template <> struct compare_op_name<CompareOp::IsNegativeInf>
{
    static constexpr char value[] = "IsNegativeInf";
};
template <> struct compare_op_name<CompareOp::PerChannel>
{
    static constexpr char value[] = "PerChannel";
};
template <> struct compare_op_name<CompareOp::AnyChannel>
{
    static constexpr char value[] = "AnyChannel";
};

inline std::ostream &operator<<(std::ostream &aOs, const CompareOp &aCompareOp)
{
    switch (CompareOp_NoFlags(aCompareOp))
    {
        case mpp::CompareOp::Less:
            aOs << compare_op_name<CompareOp::Less>::value;
            break;
        case mpp::CompareOp::LessEq:
            aOs << compare_op_name<CompareOp::LessEq>::value;
            break;
        case mpp::CompareOp::Eq:
            aOs << compare_op_name<CompareOp::Eq>::value;
            break;
        case mpp::CompareOp::Greater:
            aOs << compare_op_name<CompareOp::Greater>::value;
            break;
        case mpp::CompareOp::GreaterEq:
            aOs << compare_op_name<CompareOp::GreaterEq>::value;
            break;
        case mpp::CompareOp::NEq:
            aOs << compare_op_name<CompareOp::NEq>::value;
            break;
        case mpp::CompareOp::IsFinite:
            aOs << compare_op_name<CompareOp::IsFinite>::value;
            break;
        case mpp::CompareOp::IsNaN:
            aOs << compare_op_name<CompareOp::IsNaN>::value;
            break;
        case mpp::CompareOp::IsInf:
            aOs << compare_op_name<CompareOp::IsInf>::value;
            break;
        case mpp::CompareOp::IsInfOrNaN:
            aOs << compare_op_name<CompareOp::IsInfOrNaN>::value;
            break;
        case mpp::CompareOp::IsPositiveInf:
            aOs << compare_op_name<CompareOp::IsPositiveInf>::value;
            break;
        case mpp::CompareOp::IsNegativeInf:
            aOs << compare_op_name<CompareOp::IsNegativeInf>::value;
            break;
        default:
        {
            const std::ios::fmtflags f(aOs.flags());
            aOs << "Unknown: 0x" << std::hex << std::uppercase << static_cast<uint>(aCompareOp);
            aOs.flags(f);
        }
        break;
    }
    if (CompareOp_IsAnyChannel(aCompareOp))
    {
        aOs << " | " << compare_op_name<CompareOp::AnyChannel>::value;
    }
    if (CompareOp_IsPerChannel(aCompareOp))
    {
        aOs << " | " << compare_op_name<CompareOp::PerChannel>::value;
    }
    return aOs;
}

inline std::wostream &operator<<(std::wostream &aOs, const CompareOp &aCompareOp)
{
    switch (CompareOp_NoFlags(aCompareOp))
    {
        case mpp::CompareOp::Less:
            aOs << compare_op_name<CompareOp::Less>::value;
            break;
        case mpp::CompareOp::LessEq:
            aOs << compare_op_name<CompareOp::LessEq>::value;
            break;
        case mpp::CompareOp::Eq:
            aOs << compare_op_name<CompareOp::Eq>::value;
            break;
        case mpp::CompareOp::Greater:
            aOs << compare_op_name<CompareOp::Greater>::value;
            break;
        case mpp::CompareOp::GreaterEq:
            aOs << compare_op_name<CompareOp::GreaterEq>::value;
            break;
        case mpp::CompareOp::NEq:
            aOs << compare_op_name<CompareOp::NEq>::value;
            break;
        case mpp::CompareOp::IsFinite:
            aOs << compare_op_name<CompareOp::IsFinite>::value;
            break;
        case mpp::CompareOp::IsNaN:
            aOs << compare_op_name<CompareOp::IsNaN>::value;
            break;
        case mpp::CompareOp::IsInf:
            aOs << compare_op_name<CompareOp::IsInf>::value;
            break;
        case mpp::CompareOp::IsInfOrNaN:
            aOs << compare_op_name<CompareOp::IsInfOrNaN>::value;
            break;
        case mpp::CompareOp::IsPositiveInf:
            aOs << compare_op_name<CompareOp::IsPositiveInf>::value;
            break;
        case mpp::CompareOp::IsNegativeInf:
            aOs << compare_op_name<CompareOp::IsNegativeInf>::value;
            break;
        default:
        {
            const std::ios::fmtflags f(aOs.flags());
            aOs << "Unknown: 0x" << std::hex << std::uppercase << static_cast<uint>(aCompareOp);
            aOs.flags(f);
        }
        break;
    }
    if (CompareOp_IsAnyChannel(aCompareOp))
    {
        aOs << " | " << compare_op_name<CompareOp::AnyChannel>::value;
    }
    if (CompareOp_IsPerChannel(aCompareOp))
    {
        aOs << " | " << compare_op_name<CompareOp::PerChannel>::value;
    }
    return aOs;
}
#pragma endregion

#pragma region BorderType
/// <summary>
/// Border modes for image filtering<para/>
/// Note: NPP currently only supports NPP_BORDER_REPLICATE, why we will base the enum values on IPP instead:
/// </summary>
enum class BorderType // NOLINT(performance-enum-size)
{
    /// <summary>
    /// Undefined image border type.
    /// </summary>
    None,
    /// <summary>
    /// For a 4-pixel image | 0 1 2 3 | borders are treated as in:<para/>
    /// c c c | 0 1 2 3 | c c c for a constant c
    /// </summary>
    Constant,
    /// <summary>
    /// For a 4-pixel image | 0 1 2 3 | borders are treated as in:<para/>
    /// 0 0 0 | 0 1 2 3 | 3 3 3
    /// </summary>
    Replicate,
    /// <summary>
    /// For a 4-pixel image | 0 1 2 3 | borders are treated as in:<para/>
    /// 3 2 1 | 0 1 2 3 | 2 1 0
    /// </summary>
    Mirror,
    /// <summary>
    /// For a 4-pixel image | 0 1 2 3 | borders are treated as in:<para/>
    /// 2 1 0 | 0 1 2 3 | 3 2 1
    /// </summary>
    MirrorReplicate,
    /// <summary>
    /// For a 4-pixel image | 0 1 2 3 | borders are treated as in:<para/>
    /// 1 2 3 | 0 1 2 3 | 0 1 2
    /// </summary>
    Wrap,
    /// <summary>
    /// In SmoothEdge border type, all pixels that fall outside the input image ROI are ignored and the destination
    /// pixel is not written. Except for the one pixel line sourrounding the input ROI, here the image pixels are
    /// extrapolated in order to obtain "a smooth edge". It is not exactly the same algorithm as in IPP, but similar.
    /// Note: In NPP and IPP, SmoothEdge is a flag for interpolation mode, but a member in BorderType seems more
    /// reasonable...
    /// </summary>
    SmoothEdge
};

template <BorderType T> struct border_type_name
{
    static constexpr char value[] = "Unknown";
};
template <> struct border_type_name<BorderType::None>
{
    static constexpr char value[] = "None";
};
template <> struct border_type_name<BorderType::Constant>
{
    static constexpr char value[] = "Constant";
};
template <> struct border_type_name<BorderType::Replicate>
{
    static constexpr char value[] = "Replicate";
};
template <> struct border_type_name<BorderType::Mirror>
{
    static constexpr char value[] = "Mirror";
};
template <> struct border_type_name<BorderType::MirrorReplicate>
{
    static constexpr char value[] = "MirrorReplicate";
};
template <> struct border_type_name<BorderType::Wrap>
{
    static constexpr char value[] = "Wrap";
};
template <> struct border_type_name<BorderType::SmoothEdge>
{
    static constexpr char value[] = "SmoothEdge";
};

inline std::ostream &operator<<(std::ostream &aOs, const BorderType &aBorderType)
{
    switch (aBorderType)
    {
        case mpp::BorderType::None:
            aOs << border_type_name<BorderType::None>::value;
            break;
        case mpp::BorderType::Constant:
            aOs << border_type_name<BorderType::Constant>::value;
            break;
        case mpp::BorderType::Replicate:
            aOs << border_type_name<BorderType::Replicate>::value;
            break;
        case mpp::BorderType::Mirror:
            aOs << border_type_name<BorderType::Mirror>::value;
            break;
        case mpp::BorderType::MirrorReplicate:
            aOs << border_type_name<BorderType::MirrorReplicate>::value;
            break;
        case mpp::BorderType::Wrap:
            aOs << border_type_name<BorderType::Wrap>::value;
            break;
        case mpp::BorderType::SmoothEdge:
            aOs << border_type_name<BorderType::SmoothEdge>::value;
            break;
        default:
        {
            const std::ios::fmtflags f(aOs.flags());
            aOs << "Unknown: 0x" << std::hex << std::uppercase << static_cast<uint>(aBorderType);
            aOs.flags(f);
        }
        break;
    }
    return aOs;
}

inline std::wostream &operator<<(std::wostream &aOs, const BorderType &aBorderType)
{
    switch (aBorderType)
    {
        case mpp::BorderType::None:
            aOs << border_type_name<BorderType::None>::value;
            break;
        case mpp::BorderType::Constant:
            aOs << border_type_name<BorderType::Constant>::value;
            break;
        case mpp::BorderType::Replicate:
            aOs << border_type_name<BorderType::Replicate>::value;
            break;
        case mpp::BorderType::Mirror:
            aOs << border_type_name<BorderType::Mirror>::value;
            break;
        case mpp::BorderType::MirrorReplicate:
            aOs << border_type_name<BorderType::MirrorReplicate>::value;
            break;
        case mpp::BorderType::Wrap:
            aOs << border_type_name<BorderType::Wrap>::value;
            break;
        case mpp::BorderType::SmoothEdge:
            aOs << border_type_name<BorderType::SmoothEdge>::value;
            break;
        default:
        {
            const std::ios::fmtflags f(aOs.flags());
            aOs << "Unknown: 0x" << std::hex << std::uppercase << static_cast<uint>(aBorderType);
            aOs.flags(f);
        }
        break;
    }
    return aOs;
}
#pragma endregion

#pragma region HistorgamEvenMode
/// <summary>
/// Defines how to evenly spread an integer distribution<para/>
/// NPP uses a different definition on how two create evenly spaced bins for histograms for integer data. MPP supports
/// both definitions in its EvenLevels function.
/// </summary>
enum class HistorgamEvenMode // NOLINT(performance-enum-size)
{
    /// <summary>
    /// Tries to reproduce the same even distribution as used in the CUB implementiation (used as backend for
    /// HistogramEven)
    /// </summary>
    Default,
    /// <summary>
    /// Tries to reproduce the same even distribution as used in the NPP implementation of EvenLevels and HistogramEven
    /// </summary>
    NPP
};

template <HistorgamEvenMode T> struct historgam_even_mode_name
{
    static constexpr char value[] = "Unknown";
};
template <> struct historgam_even_mode_name<HistorgamEvenMode::Default>
{
    static constexpr char value[] = "Default";
};
template <> struct historgam_even_mode_name<HistorgamEvenMode::NPP>
{
    static constexpr char value[] = "NPP";
};

inline std::ostream &operator<<(std::ostream &aOs, const HistorgamEvenMode &aHistorgamEvenMode)
{
    switch (aHistorgamEvenMode)
    {
        case mpp::HistorgamEvenMode::Default:
            aOs << historgam_even_mode_name<HistorgamEvenMode::Default>::value;
            break;
        case mpp::HistorgamEvenMode::NPP:
            aOs << historgam_even_mode_name<HistorgamEvenMode::NPP>::value;
            break;
        default:
        {
            const std::ios::fmtflags f(aOs.flags());
            aOs << "Unknown: 0x" << std::hex << std::uppercase << static_cast<uint>(aHistorgamEvenMode);
            aOs.flags(f);
        }
        break;
    }
    return aOs;
}

inline std::wostream &operator<<(std::wostream &aOs, const HistorgamEvenMode &aHistorgamEvenMode)
{
    switch (aHistorgamEvenMode)
    {
        case mpp::HistorgamEvenMode::Default:
            aOs << historgam_even_mode_name<HistorgamEvenMode::Default>::value;
            break;
        case mpp::HistorgamEvenMode::NPP:
            aOs << historgam_even_mode_name<HistorgamEvenMode::NPP>::value;
            break;
        default:
        {
            const std::ios::fmtflags f(aOs.flags());
            aOs << "Unknown: 0x" << std::hex << std::uppercase << static_cast<uint>(aHistorgamEvenMode);
            aOs.flags(f);
        }
        break;
    }
    return aOs;
}
#pragma endregion

#pragma region InterpolationMode
/// <summary>
/// Pixel interpolation modes
/// </summary>
enum class InterpolationMode // NOLINT(performance-enum-size)
{
    /// <summary>
    /// Undefined interpolation mode
    /// </summary>
    Undefined = 0,
    /// <summary>
    /// Nearest Neighbor interpolation mode
    /// </summary>
    NearestNeighbor = 1,
    /// <summary>
    /// Bi-Linear interpolation mode
    /// </summary>
    Linear = 2,
    /// <summary>
    /// Bi-Cubic interpolation mode using Cubic Hermite Splines, named 'cubic' in Matlab
    /// </summary>
    CubicHermiteSpline = 3,
    /// <summary>
    /// Bi-Cubic interpolation mode using Lagrange polynomials, named 'Cubic' in NPP (and IPP?)
    /// </summary>
    CubicLagrange = 4,
    /// <summary>
    /// Bi-Cubic interpolation mode with two-parameter cubic filter (B=1, C=0)
    /// </summary>
    Cubic2ParamBSpline = 5,
    /// <summary>
    /// Bi-Cubic interpolation mode with two-parameter cubic filter (B=0, C=1/2)
    /// </summary>
    Cubic2ParamCatmullRom = 6,
    /// <summary>
    /// Bi-Cubic interpolation mode with two-parameter cubic filter (B=1/2, C=3/10)
    /// </summary>
    Cubic2ParamB05C03 = 7,
    /// <summary>
    /// Super Sampling interpolation mode<para/>
    /// Note: The super sampling interpolation mode can only be used if width and height of the destination image are
    /// smaller than the source image.
    /// </summary>
    Super = 8,
    /// <summary>
    /// Interpolation with the 2-lobed Lanczos Window Function<para/>
    /// The interpolation algorithm uses source image intensities at 16 pixels in the neighborhood of the point in the
    /// source image.
    /// </summary>
    Lanczos2Lobed = 9,
    /// <summary>
    /// Interpolation with the 3-lobed Lanczos Window Function<para/>
    /// The interpolation algorithm uses source image intensities at 36 pixels in the neighborhood of the point in the
    /// source image.
    /// Note: This is the same as Lanczos in NPP.
    /// </summary>
    Lanczos3Lobed = 10
};

template <InterpolationMode T> struct interpolation_mode_name
{
    static constexpr char value[] = "Unknown";
};
template <> struct interpolation_mode_name<InterpolationMode::Undefined>
{
    static constexpr char value[] = "Undefined";
};
template <> struct interpolation_mode_name<InterpolationMode::NearestNeighbor>
{
    static constexpr char value[] = "NearestNeighbor";
};
template <> struct interpolation_mode_name<InterpolationMode::Linear>
{
    static constexpr char value[] = "Linear";
};
template <> struct interpolation_mode_name<InterpolationMode::CubicHermiteSpline>
{
    static constexpr char value[] = "CubicHermiteSpline";
};
template <> struct interpolation_mode_name<InterpolationMode::CubicLagrange>
{
    static constexpr char value[] = "CubicLagrange";
};
template <> struct interpolation_mode_name<InterpolationMode::Cubic2ParamBSpline>
{
    static constexpr char value[] = "Cubic2ParamBSpline";
};
template <> struct interpolation_mode_name<InterpolationMode::Cubic2ParamCatmullRom>
{
    static constexpr char value[] = "Cubic2ParamCatmullRom";
};
template <> struct interpolation_mode_name<InterpolationMode::Cubic2ParamB05C03>
{
    static constexpr char value[] = "Cubic2ParamB05C03";
};
template <> struct interpolation_mode_name<InterpolationMode::Super>
{
    static constexpr char value[] = "Super";
};
template <> struct interpolation_mode_name<InterpolationMode::Lanczos2Lobed>
{
    static constexpr char value[] = "Lanczos2Lobed";
};
template <> struct interpolation_mode_name<InterpolationMode::Lanczos3Lobed>
{
    static constexpr char value[] = "Lanczos3Lobed";
};

inline std::ostream &operator<<(std::ostream &aOs, const InterpolationMode &aInterpolationMode)
{
    switch (aInterpolationMode)
    {
        case mpp::InterpolationMode::Undefined:
            aOs << interpolation_mode_name<InterpolationMode::Undefined>::value;
            break;
        case mpp::InterpolationMode::NearestNeighbor:
            aOs << interpolation_mode_name<InterpolationMode::NearestNeighbor>::value;
            break;
        case mpp::InterpolationMode::Linear:
            aOs << interpolation_mode_name<InterpolationMode::Linear>::value;
            break;
        case mpp::InterpolationMode::CubicHermiteSpline:
            aOs << interpolation_mode_name<InterpolationMode::CubicHermiteSpline>::value;
            break;
        case mpp::InterpolationMode::CubicLagrange:
            aOs << interpolation_mode_name<InterpolationMode::CubicLagrange>::value;
            break;
        case mpp::InterpolationMode::Cubic2ParamBSpline:
            aOs << interpolation_mode_name<InterpolationMode::Cubic2ParamBSpline>::value;
            break;
        case mpp::InterpolationMode::Cubic2ParamCatmullRom:
            aOs << interpolation_mode_name<InterpolationMode::Cubic2ParamCatmullRom>::value;
            break;
        case mpp::InterpolationMode::Cubic2ParamB05C03:
            aOs << interpolation_mode_name<InterpolationMode::Cubic2ParamB05C03>::value;
            break;
        case mpp::InterpolationMode::Super:
            aOs << interpolation_mode_name<InterpolationMode::Super>::value;
            break;
        case mpp::InterpolationMode::Lanczos2Lobed:
            aOs << interpolation_mode_name<InterpolationMode::Lanczos2Lobed>::value;
            break;
        case mpp::InterpolationMode::Lanczos3Lobed:
            aOs << interpolation_mode_name<InterpolationMode::Lanczos3Lobed>::value;
            break;
        default:
        {
            const std::ios::fmtflags f(aOs.flags());
            aOs << "Unknown: 0x" << std::hex << std::uppercase << static_cast<uint>(aInterpolationMode);
            aOs.flags(f);
        }
        break;
    }
    return aOs;
}

inline std::wostream &operator<<(std::wostream &aOs, const InterpolationMode &aInterpolationMode)
{
    switch (aInterpolationMode)
    {
        case mpp::InterpolationMode::Undefined:
            aOs << interpolation_mode_name<InterpolationMode::Undefined>::value;
            break;
        case mpp::InterpolationMode::NearestNeighbor:
            aOs << interpolation_mode_name<InterpolationMode::NearestNeighbor>::value;
            break;
        case mpp::InterpolationMode::Linear:
            aOs << interpolation_mode_name<InterpolationMode::Linear>::value;
            break;
        case mpp::InterpolationMode::CubicHermiteSpline:
            aOs << interpolation_mode_name<InterpolationMode::CubicHermiteSpline>::value;
            break;
        case mpp::InterpolationMode::CubicLagrange:
            aOs << interpolation_mode_name<InterpolationMode::CubicLagrange>::value;
            break;
        case mpp::InterpolationMode::Cubic2ParamBSpline:
            aOs << interpolation_mode_name<InterpolationMode::Cubic2ParamBSpline>::value;
            break;
        case mpp::InterpolationMode::Cubic2ParamCatmullRom:
            aOs << interpolation_mode_name<InterpolationMode::Cubic2ParamCatmullRom>::value;
            break;
        case mpp::InterpolationMode::Cubic2ParamB05C03:
            aOs << interpolation_mode_name<InterpolationMode::Cubic2ParamB05C03>::value;
            break;
        case mpp::InterpolationMode::Super:
            aOs << interpolation_mode_name<InterpolationMode::Super>::value;
            break;
        case mpp::InterpolationMode::Lanczos2Lobed:
            aOs << interpolation_mode_name<InterpolationMode::Lanczos2Lobed>::value;
            break;
        case mpp::InterpolationMode::Lanczos3Lobed:
            aOs << interpolation_mode_name<InterpolationMode::Lanczos3Lobed>::value;
            break;
        default:
        {
            const std::ios::fmtflags f(aOs.flags());
            aOs << "Unknown: 0x" << std::hex << std::uppercase << static_cast<uint>(aInterpolationMode);
            aOs.flags(f);
        }
        break;
    }
    return aOs;
}
#pragma endregion

#pragma region MaskSize
/// <summary>
/// Mask sizes for fixed size filters
/// </summary>
enum class MaskSize // NOLINT(performance-enum-size)
{
    /// <summary>
    /// 1 x 3 filter mask.
    /// </summary>
    Mask_1x3,
    /// <summary>
    /// 1 x 5 filter mask.
    /// </summary>
    Mask_1x5,
    /// <summary>
    /// 3 x 1 filter mask.
    /// </summary>
    Mask_3x1 = 100,
    /// <summary>
    /// 5 x 1 filter mask.
    /// </summary>
    Mask_5x1,
    /// <summary>
    /// 3 x 3 filter mask.
    /// </summary>
    Mask_3x3 = 200,
    /// <summary>
    /// 5 x 5 filter mask.
    /// </summary>
    Mask_5x5,
    /// <summary>
    /// 7 x 7 filter mask.
    /// </summary>
    Mask_7x7 = 400,
    /// <summary>
    /// 9 x 9 filter mask.
    /// </summary>
    Mask_9x9 = 500,
    /// <summary>
    /// 11 x 11 filter mask.
    /// </summary>
    Mask_11x11 = 600,
    /// <summary>
    /// 13 x 13 filter mask.
    /// </summary>
    Mask_13x13 = 700,
    /// <summary>
    /// 15 x 15 filter mask.
    /// </summary>
    Mask_15x15 = 800
};

template <MaskSize T> struct mask_size_name
{
    static constexpr char value[] = "Unknown";
};
template <> struct mask_size_name<MaskSize::Mask_1x3>
{
    static constexpr char value[] = "Mask 1 x 3";
};
template <> struct mask_size_name<MaskSize::Mask_1x5>
{
    static constexpr char value[] = "Mask 1 x 5";
};
template <> struct mask_size_name<MaskSize::Mask_3x1>
{
    static constexpr char value[] = "Mask 3 x 1";
};
template <> struct mask_size_name<MaskSize::Mask_5x1>
{
    static constexpr char value[] = "Mask 5 x 1";
};
template <> struct mask_size_name<MaskSize::Mask_3x3>
{
    static constexpr char value[] = "Mask 3 x 3";
};
template <> struct mask_size_name<MaskSize::Mask_5x5>
{
    static constexpr char value[] = "Mask 5 x 5";
};
template <> struct mask_size_name<MaskSize::Mask_7x7>
{
    static constexpr char value[] = "Mask 7 x 7";
};
template <> struct mask_size_name<MaskSize::Mask_9x9>
{
    static constexpr char value[] = "Mask 9 x 9";
};
template <> struct mask_size_name<MaskSize::Mask_11x11>
{
    static constexpr char value[] = "Mask 11 x 11";
};
template <> struct mask_size_name<MaskSize::Mask_13x13>
{
    static constexpr char value[] = "Mask 13 x 13";
};
template <> struct mask_size_name<MaskSize::Mask_15x15>
{
    static constexpr char value[] = "Mask 15 x 15";
};

inline std::ostream &operator<<(std::ostream &aOs, const MaskSize &aMaskSize)
{
    switch (aMaskSize)
    {
        case mpp::MaskSize::Mask_1x3:
            aOs << mask_size_name<MaskSize::Mask_1x3>::value;
            break;
        case mpp::MaskSize::Mask_1x5:
            aOs << mask_size_name<MaskSize::Mask_1x5>::value;
            break;
        case mpp::MaskSize::Mask_3x1:
            aOs << mask_size_name<MaskSize::Mask_3x1>::value;
            break;
        case mpp::MaskSize::Mask_5x1:
            aOs << mask_size_name<MaskSize::Mask_5x1>::value;
            break;
        case mpp::MaskSize::Mask_3x3:
            aOs << mask_size_name<MaskSize::Mask_3x3>::value;
            break;
        case mpp::MaskSize::Mask_5x5:
            aOs << mask_size_name<MaskSize::Mask_5x5>::value;
            break;
        case mpp::MaskSize::Mask_7x7:
            aOs << mask_size_name<MaskSize::Mask_7x7>::value;
            break;
        case mpp::MaskSize::Mask_9x9:
            aOs << mask_size_name<MaskSize::Mask_9x9>::value;
            break;
        case mpp::MaskSize::Mask_11x11:
            aOs << mask_size_name<MaskSize::Mask_11x11>::value;
            break;
        case mpp::MaskSize::Mask_13x13:
            aOs << mask_size_name<MaskSize::Mask_13x13>::value;
            break;
        case mpp::MaskSize::Mask_15x15:
            aOs << mask_size_name<MaskSize::Mask_15x15>::value;
            break;
        default:
        {
            const std::ios::fmtflags f(aOs.flags());
            aOs << "Unknown: 0x" << std::hex << std::uppercase << static_cast<uint>(aMaskSize);
            aOs.flags(f);
        }
        break;
    }
    return aOs;
}

inline std::wostream &operator<<(std::wostream &aOs, const MaskSize &aMaskSize)
{
    switch (aMaskSize)
    {
        case mpp::MaskSize::Mask_1x3:
            aOs << mask_size_name<MaskSize::Mask_1x3>::value;
            break;
        case mpp::MaskSize::Mask_1x5:
            aOs << mask_size_name<MaskSize::Mask_1x5>::value;
            break;
        case mpp::MaskSize::Mask_3x1:
            aOs << mask_size_name<MaskSize::Mask_3x1>::value;
            break;
        case mpp::MaskSize::Mask_5x1:
            aOs << mask_size_name<MaskSize::Mask_5x1>::value;
            break;
        case mpp::MaskSize::Mask_3x3:
            aOs << mask_size_name<MaskSize::Mask_3x3>::value;
            break;
        case mpp::MaskSize::Mask_5x5:
            aOs << mask_size_name<MaskSize::Mask_5x5>::value;
            break;
        case mpp::MaskSize::Mask_7x7:
            aOs << mask_size_name<MaskSize::Mask_7x7>::value;
            break;
        case mpp::MaskSize::Mask_9x9:
            aOs << mask_size_name<MaskSize::Mask_9x9>::value;
            break;
        case mpp::MaskSize::Mask_11x11:
            aOs << mask_size_name<MaskSize::Mask_11x11>::value;
            break;
        case mpp::MaskSize::Mask_13x13:
            aOs << mask_size_name<MaskSize::Mask_13x13>::value;
            break;
        case mpp::MaskSize::Mask_15x15:
            aOs << mask_size_name<MaskSize::Mask_15x15>::value;
            break;
        default:
        {
            const std::ios::fmtflags f(aOs.flags());
            aOs << "Unknown: 0x" << std::hex << std::uppercase << static_cast<uint>(aMaskSize);
            aOs.flags(f);
        }
        break;
    }
    return aOs;
}
#pragma endregion

#pragma region FixedFilter
/// <summary>
/// Filters with a fixed coeffient matrix
/// </summary>
enum class FixedFilter // NOLINT(performance-enum-size)
{
    /// <summary>
    /// Gauss filter. Possible mask sizes: 3x3, 5x5, 7x7, 9x9, 11x11, 13x13 and 15x15.<para/>
    /// Filter kernels for Gauss filter are calculated using a sigma value of 0.4 + (filter width / 3.0) * 0.6.<para/>
    /// Note: In NPP the sigma value is given as 0.4 + (filter width / 2) * 0.6, but the values actually used are "width
    /// / 3". Further, the kernel values are normalized to get an exact sum equal to 1.
    /// </summary>
    Gauss,
    /// <summary>
    /// High pass filter. Possible mask sizes: 3x3 and 5x5.<para/>
    /// Used filters are:<para/>
    /// -1 -1 -1<para/>
    /// -1 8 -1<para/>
    /// -1 -1 -1<para/>
    /// and<para/>
    /// -1 -1 -1 -1 -1<para/>
    /// -1 -1 -1 -1 -1<para/>
    /// -1 -1 24 -1 -1<para/>
    /// -1 -1 -1 -1 -1<para/>
    /// -1 -1 -1 -1 -1<para/>
    /// </summary>
    HighPass,
    /// <summary>
    /// Low pass filter. Possible mask sizes: 3x3 and 5x5.<para/>
    /// Used filters are:<para/>
    /// 1/9 1/9 1/9<para/>
    /// 1/9 1/9 1/9<para/>
    /// 1/9 1/9 1/9<para/>
    /// and<para/>
    /// 1/25 1/25 1/25 1/25 1/25 <para/>
    /// 1/25 1/25 1/25 1/25 1/25 <para/>
    /// 1/25 1/25 1/25 1/25 1/25 <para/>
    /// 1/25 1/25 1/25 1/25 1/25 <para/>
    /// 1/25 1/25 1/25 1/25 1/25 <para/>
    /// </summary>
    LowPass,
    /// <summary>
    /// Laplace filter. Possible mask sizes: 3x3 and 5x5.<para/>
    /// Used filters are:<para/>
    /// -1 -1 -1<para/>
    /// -1 8 -1<para/>
    /// -1 -1 -1<para/>
    /// and<para/>
    /// -1 -3 -4 -3 -1<para/>
    /// -3  0  6  0 -3<para/>
    /// -4  6 20  6 -4<para/>
    /// -3  0  6  0 -3<para/>
    /// -1 -3 -4 -3 -1<para/>
    /// </summary>
    Laplace,
    /// <summary>
    /// Horizontal Prewitt filter. Possible mask size: 3x3.<para/>
    /// Used filter is:<para/>
    /// -1 -1 -1<para/>
    ///  0  0  0<para/>
    ///  1  1  1<para/>
    /// Note: the filter is mirrored compared to NPP due to different filtering logic (convolution vs correlation). MPP
    /// uses correlation through out all filtering alike algorithms and in order to obtain the same output as in NPP,
    /// the filter coefficients had to be mirrored.
    /// </summary>
    PrewittHoriz,
    /// <summary>
    /// Vertical Prewitt filter. Possible mask size: 3x3.<para/>
    /// Used filter is:<para/>
    /// -1 0 1<para/>
    /// -1 0 1<para/>
    /// -1 0 1<para/>
    /// Note: The documentation in NPP differs to the actually implemented filter. Nevertheless, this is the filter used
    /// with correlation filtering logic and gives the same results as NPP. MPP uses correlation through out all
    /// filtering alike algorithms and in order to obtain the same output as in NPP, the filter coefficients had to be
    /// mirrored.
    /// </summary>
    PrewittVert,
    /// <summary>
    /// Roberts down filter. Possible mask size: 3x3.<para/>
    /// Used filter is:<para/>
    /// -1 0 0<para/>
    ///  0 1 0<para/>
    ///  0 0 0<para/>
    /// Note: the filter is mirrored compared to NPP due to different filtering logic (convolution vs correlation). MPP
    /// uses correlation through out all filtering alike algorithms and in order to obtain the same output as in NPP,
    /// the filter coefficients had to be mirrored.
    /// </summary>
    RobertsDown,
    /// <summary>
    /// Roberts up filter. Possible mask size: 3x3.<para/>
    /// Used filter is:<para/>
    /// 0 0 -1<para/>
    /// 0 1  0<para/>
    /// 0 0  0<para/>
    /// Note: the filter is mirrored compared to NPP due to different filtering logic (convolution vs correlation). MPP
    /// uses correlation through out all filtering alike algorithms and in order to obtain the same output as in NPP,
    /// the filter coefficients had to be mirrored.
    /// </summary>
    RobertsUp,
    /// <summary>
    /// Horizontal Scharr filter. Possible mask size: 3x3.<para/>
    /// Used filter is:<para/>
    /// -3 -10 -3<para/>
    ///  0   0  0<para/>
    ///  3  10  3<para/>
    /// Note: the filter is mirrored compared to NPP due to different filtering logic (convolution vs correlation). MPP
    /// uses correlation through out all filtering alike algorithms and in order to obtain the same output as in NPP,
    /// the filter coefficients had to be mirrored.
    /// </summary>
    ScharrHoriz,
    /// <summary>
    /// Vertical Prewitt filter. Possible mask size: 3x3.<para/>
    /// Used filter is:<para/>
    ///  -3 0 3<para/>
    /// -10 0 10<para/>
    ///  -3 0 3<para/>
    /// Note: The documentation in NPP differs to the actually implemented filter. Nevertheless, this is the filter used
    /// with correlation filtering logic and gives the same results as NPP. MPP uses correlation through out all
    /// filtering alike algorithms and in order to obtain the same output as in NPP, the filter coefficients had to be
    /// mirrored.
    /// </summary>
    ScharrVert,
    /// <summary>
    /// Sharpen filter. Possible mask size: 3x3.<para/>
    /// Used filter is:<para/>
    /// -1/8 -1/8 -1/8<para/>
    /// -1/8 16/8 -1/8<para/>
    /// -1/8 -1/8 -1/8<para/>
    /// </summary>
    Sharpen,
    /// <summary>
    /// Second cross derivative Sobel filter. Possible mask sizes: 3x3 and 5x5.<para/>
    /// Used filters are:<para/>
    /// -1 0  1<para/>
    ///  0 0  0<para/>
    ///  1 0 -1<para/>
    /// and<para/>
    /// -1 -2  0  2  1<para/>
    /// -2 -4  0  4  2<para/>
    ///  0  0  0  0  0<para/>
    ///  2  4  0 -4 -2<para/>
    ///  1  2  0 -2 -1<para/>
    /// </summary>
    SobelCross,
    /// <summary>
    /// Horizontal Sobel filter. Possible mask sizes: 3x3 and 5x5.<para/>
    /// Used filters are:<para/>
    ///  1  2  1<para/>
    ///  0  0  0<para/>
    /// -1 -2 -1<para/>
    /// and<para/>
    /// -1 -4  -6 -4 -1<para/>
    /// -2 -8 -12 -8 -2<para/>
    ///  0  0   0  0  0<para/>
    ///  2  8  12  8  2<para/>
    ///  1  4   6  4  1<para/>
    /// Note: the filter is mirrored compared to NPP due to different filtering logic (convolution vs correlation). MPP
    /// uses correlation through out all filtering alike algorithms and in order to obtain the same output as in NPP,
    /// the filter coefficients had to be mirrored.
    /// </summary>
    SobelHoriz,
    /// <summary>
    /// Vertical Sobel filter. Possible mask sizes: 3x3 and 5x5.<para/>
    /// Used filters are:<para/>
    /// 1 0 -1<para/>
    /// 2 0 -2<para/>
    /// 1 0 -1<para/>
    /// and<para/>
    /// -1  -2 0  2 1<para/>
    /// -4  -8 0  8 4<para/>
    /// -6 -12 0 12 6<para/>
    /// -4  -8 0  8 4<para/>
    /// -1  -2 0  2 1<para/>
    /// Note: The documentation in NPP differs to the actually implemented filter. Nevertheless, this is the filter used
    /// with correlation filtering logic and gives the same results as NPP. MPP uses correlation through out all
    /// filtering alike algorithms and in order to obtain the same output as in NPP, the filter coefficients had to be
    /// mirrored.
    /// </summary>
    SobelVert,
    /// <summary>
    /// Second derivative horizontal Sobel filter. Possible mask sizes: 3x3 and 5x5.<para/>
    /// Used filters are:<para/>
    ///  1  2  1<para/>
    /// -2 -4 -2<para/>
    ///  1  2  1<para/>
    /// and<para/>
    ///  1  4   6  4  1<para/>
    ///  0  0   0  0  0<para/>
    /// -2 -8 -12 -8 -2<para/>
    ///  0  0   0  0  0<para/>
    ///  1  4   6  4  1<para/>
    /// </summary>
    SobelHorizSecond,
    /// <summary>
    /// Second derivative vertical Sobel filter. Possible mask sizes: 3x3 and 5x5.<para/>
    /// Used filters are:<para/>
    /// 1 -2 1<para/>
    /// 2  4 2<para/>
    /// 1 -2 1<para/>
    /// and<para/>
    /// 1  0  -2  0  1<para/>
    /// 4  0  -8  0  4<para/>
    /// 6  0 -12  0  6<para/>
    /// 4  0  -8  0  4<para/>
    /// 1  0  -2  0  1<para/>
    /// </summary>
    SobelVertSecond
};

template <FixedFilter T> struct fixed_filter_name
{
    static constexpr char value[] = "Unknown";
};
template <> struct fixed_filter_name<FixedFilter::HighPass>
{
    static constexpr char value[] = "HighPass";
};
template <> struct fixed_filter_name<FixedFilter::LowPass>
{
    static constexpr char value[] = "LowPass";
};
template <> struct fixed_filter_name<FixedFilter::Laplace>
{
    static constexpr char value[] = "Laplace";
};
template <> struct fixed_filter_name<FixedFilter::PrewittHoriz>
{
    static constexpr char value[] = "PrewittHoriz";
};
template <> struct fixed_filter_name<FixedFilter::PrewittVert>
{
    static constexpr char value[] = "PrewittVert";
};
template <> struct fixed_filter_name<FixedFilter::RobertsDown>
{
    static constexpr char value[] = "RobertsDown";
};
template <> struct fixed_filter_name<FixedFilter::RobertsUp>
{
    static constexpr char value[] = "RobertsUp";
};
template <> struct fixed_filter_name<FixedFilter::ScharrHoriz>
{
    static constexpr char value[] = "ScharrHoriz";
};
template <> struct fixed_filter_name<FixedFilter::ScharrVert>
{
    static constexpr char value[] = "ScharrVert";
};
template <> struct fixed_filter_name<FixedFilter::Sharpen>
{
    static constexpr char value[] = "Sharpen";
};
template <> struct fixed_filter_name<FixedFilter::SobelCross>
{
    static constexpr char value[] = "SobelCross";
};
template <> struct fixed_filter_name<FixedFilter::SobelHoriz>
{
    static constexpr char value[] = "SobelHoriz";
};
template <> struct fixed_filter_name<FixedFilter::SobelVert>
{
    static constexpr char value[] = "SobelVert";
};
template <> struct fixed_filter_name<FixedFilter::SobelHorizSecond>
{
    static constexpr char value[] = "SobelHorizSecond";
};
template <> struct fixed_filter_name<FixedFilter::SobelVertSecond>
{
    static constexpr char value[] = "SobelVertSecond";
};

inline std::ostream &operator<<(std::ostream &aOs, const FixedFilter &aFixedFilter)
{
    switch (aFixedFilter)
    {
        case mpp::FixedFilter::HighPass:
            aOs << fixed_filter_name<FixedFilter::HighPass>::value;
            break;
        case mpp::FixedFilter::LowPass:
            aOs << fixed_filter_name<FixedFilter::LowPass>::value;
            break;
        case mpp::FixedFilter::Laplace:
            aOs << fixed_filter_name<FixedFilter::Laplace>::value;
            break;
        case mpp::FixedFilter::PrewittHoriz:
            aOs << fixed_filter_name<FixedFilter::PrewittHoriz>::value;
            break;
        case mpp::FixedFilter::PrewittVert:
            aOs << fixed_filter_name<FixedFilter::PrewittVert>::value;
            break;
        case mpp::FixedFilter::RobertsDown:
            aOs << fixed_filter_name<FixedFilter::RobertsDown>::value;
            break;
        case mpp::FixedFilter::RobertsUp:
            aOs << fixed_filter_name<FixedFilter::RobertsUp>::value;
            break;
        case mpp::FixedFilter::ScharrHoriz:
            aOs << fixed_filter_name<FixedFilter::ScharrHoriz>::value;
            break;
        case mpp::FixedFilter::ScharrVert:
            aOs << fixed_filter_name<FixedFilter::ScharrVert>::value;
            break;
        case mpp::FixedFilter::Sharpen:
            aOs << fixed_filter_name<FixedFilter::Sharpen>::value;
            break;
        case mpp::FixedFilter::SobelCross:
            aOs << fixed_filter_name<FixedFilter::SobelCross>::value;
            break;
        case mpp::FixedFilter::SobelHoriz:
            aOs << fixed_filter_name<FixedFilter::SobelHoriz>::value;
            break;
        case mpp::FixedFilter::SobelVert:
            aOs << fixed_filter_name<FixedFilter::SobelVert>::value;
            break;
        case mpp::FixedFilter::SobelHorizSecond:
            aOs << fixed_filter_name<FixedFilter::SobelHorizSecond>::value;
            break;
        case mpp::FixedFilter::SobelVertSecond:
            aOs << fixed_filter_name<FixedFilter::SobelVertSecond>::value;
            break;
        default:
        {
            const std::ios::fmtflags f(aOs.flags());
            aOs << "Unknown: 0x" << std::hex << std::uppercase << static_cast<uint>(aFixedFilter);
            aOs.flags(f);
        }
        break;
    }
    return aOs;
}

inline std::wostream &operator<<(std::wostream &aOs, const FixedFilter &aFixedFilter)
{
    switch (aFixedFilter)
    {
        case mpp::FixedFilter::HighPass:
            aOs << fixed_filter_name<FixedFilter::HighPass>::value;
            break;
        case mpp::FixedFilter::LowPass:
            aOs << fixed_filter_name<FixedFilter::LowPass>::value;
            break;
        case mpp::FixedFilter::Laplace:
            aOs << fixed_filter_name<FixedFilter::Laplace>::value;
            break;
        case mpp::FixedFilter::PrewittHoriz:
            aOs << fixed_filter_name<FixedFilter::PrewittHoriz>::value;
            break;
        case mpp::FixedFilter::PrewittVert:
            aOs << fixed_filter_name<FixedFilter::PrewittVert>::value;
            break;
        case mpp::FixedFilter::RobertsDown:
            aOs << fixed_filter_name<FixedFilter::RobertsDown>::value;
            break;
        case mpp::FixedFilter::RobertsUp:
            aOs << fixed_filter_name<FixedFilter::RobertsUp>::value;
            break;
        case mpp::FixedFilter::ScharrHoriz:
            aOs << fixed_filter_name<FixedFilter::ScharrHoriz>::value;
            break;
        case mpp::FixedFilter::ScharrVert:
            aOs << fixed_filter_name<FixedFilter::ScharrVert>::value;
            break;
        case mpp::FixedFilter::Sharpen:
            aOs << fixed_filter_name<FixedFilter::Sharpen>::value;
            break;
        case mpp::FixedFilter::SobelCross:
            aOs << fixed_filter_name<FixedFilter::SobelCross>::value;
            break;
        case mpp::FixedFilter::SobelHoriz:
            aOs << fixed_filter_name<FixedFilter::SobelHoriz>::value;
            break;
        case mpp::FixedFilter::SobelVert:
            aOs << fixed_filter_name<FixedFilter::SobelVert>::value;
            break;
        case mpp::FixedFilter::SobelHorizSecond:
            aOs << fixed_filter_name<FixedFilter::SobelHorizSecond>::value;
            break;
        case mpp::FixedFilter::SobelVertSecond:
            aOs << fixed_filter_name<FixedFilter::SobelVertSecond>::value;
            break;
        default:
        {
            const std::ios::fmtflags f(aOs.flags());
            aOs << "Unknown: 0x" << std::hex << std::uppercase << static_cast<uint>(aFixedFilter);
            aOs.flags(f);
        }
        break;
    }
    return aOs;
}
#pragma endregion

#pragma region Norm
/// <summary>
/// Distance norm
/// </summary>
enum class Norm // NOLINT(performance-enum-size)
{
    /// <summary>
    /// Infinity norm (maximum)
    /// </summary>
    Inf,
    /// <summary>
    /// L1 norm (sum of absolute values)
    /// </summary>
    L1,
    /// <summary>
    /// L2 norm (square root of sum of squares)
    /// </summary>
    L2
};

template <Norm T> struct norm_name
{
    static constexpr char value[] = "Unknown";
};
template <> struct norm_name<Norm::Inf>
{
    static constexpr char value[] = "Inf";
};
template <> struct norm_name<Norm::L1>
{
    static constexpr char value[] = "L1";
};
template <> struct norm_name<Norm::L2>
{
    static constexpr char value[] = "L2";
};

inline std::ostream &operator<<(std::ostream &aOs, const Norm &aNorm)
{
    switch (aNorm)
    {
        case Norm::Inf:
            aOs << norm_name<Norm::Inf>::value;
            break;
        case Norm::L1:
            aOs << norm_name<Norm::L1>::value;
            break;
        case Norm::L2:
            aOs << norm_name<Norm::L2>::value;
            break;
        default:
        {
            const std::ios::fmtflags f(aOs.flags());
            aOs << "Unknown: 0x" << std::hex << std::uppercase << static_cast<uint>(aNorm);
            aOs.flags(f);
        }
        break;
    }
    return aOs;
}

inline std::wostream &operator<<(std::wostream &aOs, const Norm &aNorm)
{
    switch (aNorm)
    {
        case Norm::Inf:
            aOs << norm_name<Norm::Inf>::value;
            break;
        case Norm::L1:
            aOs << norm_name<Norm::L1>::value;
            break;
        case Norm::L2:
            aOs << norm_name<Norm::L2>::value;
            break;
        default:
        {
            const std::ios::fmtflags f(aOs.flags());
            aOs << "Unknown: 0x" << std::hex << std::uppercase << static_cast<uint>(aNorm);
            aOs.flags(f);
        }
        break;
    }
    return aOs;
}
#pragma endregion
} // namespace mpp