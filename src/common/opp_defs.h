#pragma once

namespace opp
{
/// <summary>
/// Rounding Modes<para/>
/// The enumerated rounding modes are used by a large number of OPP primitives
/// to allow the user to specify the method by which fractional values are converted
/// to integer values.
/// </summary>
enum class RoudingMode
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
    /// __float2int_rz in CUDA<para/>
    /// integer.<para/>
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
} // namespace opp