#pragma once
#include "common/defines.h"
#include "fileIOException.h"
#include <algorithm>
#include <execution>
#include <filesystem>

namespace mpp::fileIO
{

/// <summary>
/// Filetypes that we support
/// </summary>
enum class FileType // NOLINT(performance-enum-size)
{
    TIFF,
    UNKNOWN
};

/// <summary>
/// Guess the fileformat from its file name ending
/// </summary>
FileType GuessFileTypeFromEnding(const std::filesystem::path &aFileName);
constexpr const char *GetFileTypeName(FileType aFileType)
{
    switch (aFileType)
    {
        case FileType::TIFF:
            return "TIFF";
        default:
            return "UNKNOWN";
    }
}

/// <summary>
/// Base class for file reader and writer which provides a file name and endian-swap methods.
/// </summary>
class File
{
  private:
    std::filesystem::path mFileName;

    bool mIsLittleEndian;
    bool mIsMoved{false};

  protected:
    /// <summary>
    /// Indicates if the file to read or write is in little endian format
    /// </summary>
    [[nodiscard]] bool IsLittleEndian() const
    {
        return mIsLittleEndian;
    }

    void SetIsLittleEndian(bool aIsLittleEndian)
    {
        mIsLittleEndian = aIsLittleEndian;
    }

    void SetFileName(const std::filesystem::path &aFileName);

  public:
    explicit File(std::filesystem::path aFileName, bool aIsLittleEndian = true);

    virtual ~File() = default;

    File(const File &) = default;
    File(File &&)      = default;

    File &operator=(const File &) = default;

    File &operator=(File &&aOther) noexcept;

    template <typename T> static void EndianSwap(T &aData)
    {
        static_assert((sizeof(T) == 1U) || (sizeof(T) == 2U) || (sizeof(T) == 4U) || (sizeof(T) == 8U));

        // NOLINTBEGIN --> magic numbers
        switch (sizeof(T))
        {
            case 2:
            {
                ushort temp = *(reinterpret_cast<ushort *>(&aData));
                temp        = ushort(ushort(temp >> ushort(8U)) | ushort(temp << ushort(8U)));
                T *tempT    = reinterpret_cast<T *>(&temp);
                aData       = *tempT;
                break;
            }
            case 4:
            {
                uint temp = *(reinterpret_cast<uint *>(&aData));
                temp      = (temp >> 24U) | ((temp << 8U) & 0x00FF0000U) | ((temp >> 8U) & 0x0000FF00U) | (temp << 24U);
                T *tempT  = reinterpret_cast<T *>(&temp);
                aData     = *tempT;
                break;
            }
            case 8:
            {
                ulong64 temp = *(reinterpret_cast<ulong64 *>(&aData));
                temp = (temp >> 56U) | ((temp << 40U) & 0x00FF000000000000U) | ((temp << 24U) & 0x0000FF0000000000U) |
                       ((temp << 8U) & 0x000000FF00000000U) | ((temp >> 8U) & 0x00000000FF000000U) |
                       ((temp >> 24U) & 0x0000000000FF0000U) | ((temp >> 40U) & 0x000000000000FF00U) | (temp << 56U);
                T *tempT = reinterpret_cast<T *>(&temp);
                aData    = *tempT;
                break;
            }
            default:
                break;
        }
        // NOLINTEND --> magic numbers
    }
    template <class Iterator> static void EndianSwap(Iterator aFirst, Iterator aLast)
    {
        // run in parallel:
        std::for_each(EXECMODE(std::execution::par_unseq) aFirst, aLast, [](auto &aElem) { EndianSwap(aElem); });
    }

    static void EndianSwap(char *aData, size_t aElementCount, size_t aElementSize);

    static bool Exists(const std::filesystem::path &aFilename);

    [[nodiscard]] virtual FileType GetFileType() const = 0;

    [[nodiscard]] const std::filesystem::path &FileName() const
    {
        return mFileName;
    };

    static void ClearContent(const std::filesystem::path &aFileName);
};
} // namespace mpp::fileIO
