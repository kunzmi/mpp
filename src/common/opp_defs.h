#pragma once
#include <ostream>

namespace opp
{
#pragma region RoundingMode
/// <summary>
/// Rounding Modes<para/>
/// The enumerated rounding modes are used by a large number of OPP primitives
/// to allow the user to specify the method by which fractional values are converted
/// to integer values.
/// </summary>
enum class RoundingMode
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
            aOs << "Unknown";
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
            aOs << "Unknown";
            break;
    }
    return aOs;
}
#pragma endregion

#pragma region AlphaComposition

/// <summary>
/// Different Alpha compositing operations
/// </summary>
enum class AlphaCompositionOp
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
        case opp::AlphaCompositionOp::Over:
            aOs << alpha_composition_name<AlphaCompositionOp::Over>::value;
            break;
        case opp::AlphaCompositionOp::In:
            aOs << alpha_composition_name<AlphaCompositionOp::In>::value;
            break;
        case opp::AlphaCompositionOp::Out:
            aOs << alpha_composition_name<AlphaCompositionOp::Out>::value;
            break;
        case opp::AlphaCompositionOp::ATop:
            aOs << alpha_composition_name<AlphaCompositionOp::ATop>::value;
            break;
        case opp::AlphaCompositionOp::XOr:
            aOs << alpha_composition_name<AlphaCompositionOp::XOr>::value;
            break;
        case opp::AlphaCompositionOp::Plus:
            aOs << alpha_composition_name<AlphaCompositionOp::Plus>::value;
            break;
        case opp::AlphaCompositionOp::OverPremul:
            aOs << alpha_composition_name<AlphaCompositionOp::OverPremul>::value;
            break;
        case opp::AlphaCompositionOp::InPremul:
            aOs << alpha_composition_name<AlphaCompositionOp::InPremul>::value;
            break;
        case opp::AlphaCompositionOp::OutPremul:
            aOs << alpha_composition_name<AlphaCompositionOp::OutPremul>::value;
            break;
        case opp::AlphaCompositionOp::ATopPremul:
            aOs << alpha_composition_name<AlphaCompositionOp::ATopPremul>::value;
            break;
        case opp::AlphaCompositionOp::XOrPremul:
            aOs << alpha_composition_name<AlphaCompositionOp::XOrPremul>::value;
            break;
        case opp::AlphaCompositionOp::PlusPremul:
            aOs << alpha_composition_name<AlphaCompositionOp::PlusPremul>::value;
            break;
        default:
            aOs << "Unknown";
            break;
    }
    return aOs;
}

inline std::wostream &operator<<(std::wostream &aOs, const AlphaCompositionOp &aAlphaComposition)
{
    switch (aAlphaComposition)
    {
        case opp::AlphaCompositionOp::Over:
            aOs << alpha_composition_name<AlphaCompositionOp::Over>::value;
            break;
        case opp::AlphaCompositionOp::In:
            aOs << alpha_composition_name<AlphaCompositionOp::In>::value;
            break;
        case opp::AlphaCompositionOp::Out:
            aOs << alpha_composition_name<AlphaCompositionOp::Out>::value;
            break;
        case opp::AlphaCompositionOp::ATop:
            aOs << alpha_composition_name<AlphaCompositionOp::ATop>::value;
            break;
        case opp::AlphaCompositionOp::XOr:
            aOs << alpha_composition_name<AlphaCompositionOp::XOr>::value;
            break;
        case opp::AlphaCompositionOp::Plus:
            aOs << alpha_composition_name<AlphaCompositionOp::Plus>::value;
            break;
        case opp::AlphaCompositionOp::OverPremul:
            aOs << alpha_composition_name<AlphaCompositionOp::OverPremul>::value;
            break;
        case opp::AlphaCompositionOp::InPremul:
            aOs << alpha_composition_name<AlphaCompositionOp::InPremul>::value;
            break;
        case opp::AlphaCompositionOp::OutPremul:
            aOs << alpha_composition_name<AlphaCompositionOp::OutPremul>::value;
            break;
        case opp::AlphaCompositionOp::ATopPremul:
            aOs << alpha_composition_name<AlphaCompositionOp::ATopPremul>::value;
            break;
        case opp::AlphaCompositionOp::XOrPremul:
            aOs << alpha_composition_name<AlphaCompositionOp::XOrPremul>::value;
            break;
        case opp::AlphaCompositionOp::PlusPremul:
            aOs << alpha_composition_name<AlphaCompositionOp::PlusPremul>::value;
            break;
        default:
            aOs << "Unknown";
            break;
    }
    return aOs;
}
#pragma endregion

#pragma region BayerGridPosition
/// <summary>
/// Bayer grid position registration
/// </summary>
enum class BayerGridPosition
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
        case opp::BayerGridPosition::BGGR:
            aOs << bayer_grid_position_name<BayerGridPosition::BGGR>::value;
            break;
        case opp::BayerGridPosition::RGGB:
            aOs << bayer_grid_position_name<BayerGridPosition::RGGB>::value;
            break;
        case opp::BayerGridPosition::GBRG:
            aOs << bayer_grid_position_name<BayerGridPosition::GBRG>::value;
            break;
        case opp::BayerGridPosition::GRBG:
            aOs << bayer_grid_position_name<BayerGridPosition::GRBG>::value;
            break;
        default:
            aOs << "Unknown";
            break;
    }
    return aOs;
}

inline std::wostream &operator<<(std::wostream &aOs, const BayerGridPosition &aBayerGridPosition)
{
    switch (aBayerGridPosition)
    {
        case opp::BayerGridPosition::BGGR:
            aOs << bayer_grid_position_name<BayerGridPosition::BGGR>::value;
            break;
        case opp::BayerGridPosition::RGGB:
            aOs << bayer_grid_position_name<BayerGridPosition::RGGB>::value;
            break;
        case opp::BayerGridPosition::GBRG:
            aOs << bayer_grid_position_name<BayerGridPosition::GBRG>::value;
            break;
        case opp::BayerGridPosition::GRBG:
            aOs << bayer_grid_position_name<BayerGridPosition::GRBG>::value;
            break;
        default:
            aOs << "Unknown";
            break;
    }
    return aOs;
}
#pragma endregion

#pragma region MirrorAxis
/// <summary>
/// Mirror direction control
/// </summary>
enum class MirrorAxis
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
        case opp::MirrorAxis::Horizontal:
            aOs << mirror_axis_name<MirrorAxis::Horizontal>::value;
            break;
        case opp::MirrorAxis::Vertical:
            aOs << mirror_axis_name<MirrorAxis::Vertical>::value;
            break;
        case opp::MirrorAxis::Both:
            aOs << mirror_axis_name<MirrorAxis::Both>::value;
            break;
        default:
            aOs << "Unknown";
            break;
    }
    return aOs;
}

inline std::wostream &operator<<(std::wostream &aOs, const MirrorAxis &aMirrorAxis)
{
    switch (aMirrorAxis)
    {
        case opp::MirrorAxis::Horizontal:
            aOs << mirror_axis_name<MirrorAxis::Horizontal>::value;
            break;
        case opp::MirrorAxis::Vertical:
            aOs << mirror_axis_name<MirrorAxis::Vertical>::value;
            break;
        case opp::MirrorAxis::Both:
            aOs << mirror_axis_name<MirrorAxis::Both>::value;
            break;
        default:
            aOs << "Unknown";
            break;
    }
    return aOs;
}
#pragma endregion

#pragma region CompareOp
/// <summary>
/// Pixel comparison control values
/// </summary>
enum class CompareOp
{
    /// <summary>
    /// Returns true if the pixel value is &lt; than the value to compare with.
    /// </summary>
    Less,
    /// <summary>
    /// Returns true if the pixel value is &lt;= than the value to compare with.
    /// </summary>
    LessEq,
    /// <summary>
    /// Returns true if the pixel value is == than the value to compare with.
    /// </summary>
    Eq,
    /// <summary>
    /// Returns true if the pixel value is &gt; than the value to compare with.
    /// </summary>
    Greater,
    /// <summary>
    /// Returns true if the pixel value is &gt;= than the value to compare with.
    /// </summary>
    GreaterEq,
    /// <summary>
    /// Returns true if the pixel value is != than the value to compare with.
    /// </summary>
    NEq
};

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

inline std::ostream &operator<<(std::ostream &aOs, const CompareOp &aCompareOp)
{
    switch (aCompareOp)
    {
        case opp::CompareOp::Less:
            aOs << compare_op_name<CompareOp::Less>::value;
            break;
        case opp::CompareOp::LessEq:
            aOs << compare_op_name<CompareOp::LessEq>::value;
            break;
        case opp::CompareOp::Eq:
            aOs << compare_op_name<CompareOp::Eq>::value;
            break;
        case opp::CompareOp::Greater:
            aOs << compare_op_name<CompareOp::Greater>::value;
            break;
        case opp::CompareOp::GreaterEq:
            aOs << compare_op_name<CompareOp::GreaterEq>::value;
            break;
        case opp::CompareOp::NEq:
            aOs << compare_op_name<CompareOp::NEq>::value;
            break;
        default:
            aOs << "Unknown";
            break;
    }
    return aOs;
}

inline std::wostream &operator<<(std::wostream &aOs, const CompareOp &aCompareOp)
{
    switch (aCompareOp)
    {
        case opp::CompareOp::Less:
            aOs << compare_op_name<CompareOp::Less>::value;
            break;
        case opp::CompareOp::LessEq:
            aOs << compare_op_name<CompareOp::LessEq>::value;
            break;
        case opp::CompareOp::Eq:
            aOs << compare_op_name<CompareOp::Eq>::value;
            break;
        case opp::CompareOp::Greater:
            aOs << compare_op_name<CompareOp::Greater>::value;
            break;
        case opp::CompareOp::GreaterEq:
            aOs << compare_op_name<CompareOp::GreaterEq>::value;
            break;
        case opp::CompareOp::NEq:
            aOs << compare_op_name<CompareOp::NEq>::value;
            break;
        default:
            aOs << "Unknown";
            break;
    }
    return aOs;
}
#pragma endregion

#pragma region BorderType
/// <summary>
/// Border modes for image filtering<para/>
/// Note: NPP currently only supports NPP_BORDER_REPLICATE, why we will base the enum values on IPP instead:
/// </summary>
enum class BorderType
{
    /// <summary>
    /// Undefined image border type
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
    Wrap
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

inline std::ostream &operator<<(std::ostream &aOs, const BorderType &aBorderType)
{
    switch (aBorderType)
    {
        case opp::BorderType::None:
            aOs << border_type_name<BorderType::None>::value;
            break;
        case opp::BorderType::Constant:
            aOs << border_type_name<BorderType::Constant>::value;
            break;
        case opp::BorderType::Replicate:
            aOs << border_type_name<BorderType::Replicate>::value;
            break;
        case opp::BorderType::Mirror:
            aOs << border_type_name<BorderType::Mirror>::value;
            break;
        case opp::BorderType::MirrorReplicate:
            aOs << border_type_name<BorderType::MirrorReplicate>::value;
            break;
        case opp::BorderType::Wrap:
            aOs << border_type_name<BorderType::Wrap>::value;
            break;
        default:
            aOs << "Unknown";
            break;
    }
    return aOs;
}

inline std::wostream &operator<<(std::wostream &aOs, const BorderType &aBorderType)
{
    switch (aBorderType)
    {
        case opp::BorderType::None:
            aOs << border_type_name<BorderType::None>::value;
            break;
        case opp::BorderType::Constant:
            aOs << border_type_name<BorderType::Constant>::value;
            break;
        case opp::BorderType::Replicate:
            aOs << border_type_name<BorderType::Replicate>::value;
            break;
        case opp::BorderType::Mirror:
            aOs << border_type_name<BorderType::Mirror>::value;
            break;
        case opp::BorderType::MirrorReplicate:
            aOs << border_type_name<BorderType::MirrorReplicate>::value;
            break;
        case opp::BorderType::Wrap:
            aOs << border_type_name<BorderType::Wrap>::value;
            break;
        default:
            aOs << "Unknown";
            break;
    }
    return aOs;
}
#pragma endregion
} // namespace opp