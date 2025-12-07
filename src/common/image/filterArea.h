#pragma once
#include "../dllexport_common.h"
#include <common/defines.h>
#include <common/image/size2D.h>
#include <common/safeCast.h>
#include <common/vectorTypes.h>
#include <istream>
#include <ostream>

namespace mpp::image
{
/// <summary>
/// Combines filter size and center point in one struct for a simplified API
/// </summary>
struct alignas(4 * sizeof(int)) MPPEXPORT_COMMON FilterArea
{
    Size2D Size;
    Vector2<int> Center;

    FilterArea()
    {
    }

    /// <summary>
    /// Initializes the filter area to a square size and sets filter center to aSquareSize / 2 (integer division)
    /// </summary>
    FilterArea(int aSquareSize) : Size(aSquareSize), Center(aSquareSize / 2)
    {
    }

    /// <summary>
    /// Initializes the filter area to aSize and sets filter center to aSize / 2 (integer division)
    /// </summary>
    FilterArea(const Vector2<int> &aSize) : Size(aSize), Center(aSize / 2)
    {
    }

    /// <summary>
    /// Initializes the filter area to aSize and sets filter center to aCenter
    /// </summary>
    FilterArea(const Vector2<int> &aSize, const Vector2<int> &aCenter) : Size(aSize), Center(aCenter)
    {
    }

    /// <summary>
    /// returns true if the center pixel is inside the filter area and the size is &gt; 0 for both dimensions, false
    /// otherwise.
    /// </summary>
    bool CheckIfValid() const
    {
        if (Center.x < 0 || Center.y < 0 || Center.x >= Size.x || Center.y >= Size.y)
        {
            return false;
        }
        if (Size > 0)
        {
            return true;
        }
        return false;
    }

#ifdef IS_HOST_COMPILER
    friend std::ostream &operator<<(std::ostream &aOs, const FilterArea &aFilterArea)
    {
        aOs << "Size: " << aFilterArea.Size << " Center point: " << aFilterArea.Center;
        return aOs;
    }
    friend std::wostream &operator<<(std::wostream &aOs, const FilterArea &aFilterArea)
    {
        aOs << "Size: " << aFilterArea.Size << " Center point: " << aFilterArea.Center;
        return aOs;
    }
    friend std::istream &operator>>(std::istream &aIs, FilterArea &aFilterArea)
    {
        aIs >> aFilterArea.Size >> aFilterArea.Center;
        return aIs;
    }
    friend std::wistream &operator>>(std::wistream &aIs, FilterArea &aFilterArea)
    {
        aIs >> aFilterArea.Size >> aFilterArea.Center;
        return aIs;
    }
#endif

    bool operator==(const FilterArea &aOther) const
    {
        return aOther.Size == Size && aOther.Center == Center;
    }
    bool operator!=(const FilterArea &aOther) const
    {
        return aOther.Size != Size || aOther.Center != Center;
    }
};
} // namespace mpp::image