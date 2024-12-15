#pragma once

namespace opp::image
{
/// <summary>
/// Solve n x n-linear system of equations using gaussian elemination with partial pivoting
/// </summary>
/// <param name="aMatrix"></param>
/// <param name="aVec"></param>
/// <param name="aN"></param>
/// <returns></returns>
bool solve(double *aMatrix, double *aVec, int aN);

} // namespace opp::image