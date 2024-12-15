#pragma once

#include "common/defines.h"
#include "common/version.h"
#include "file.h"
#include <common/image/pixelTypes.h>
#include <common/vector2.h>
#include <common/vector3.h>
#include <filesystem>
#include <functional>
#include <memory>
#include <ostream>

namespace opp::fileIO
{
const std::string FILE_CREATED_BY = std::string("File created by ") + // NOLINT(cert-err58-cpp)
                                    OPP_PROJECT_NAME +                // NOLINT(cert-err58-cpp)
                                    " version " +                     // NOLINT(cert-err58-cpp)
                                    OPP_VERSION;                      // NOLINT(cert-err58-cpp)

constexpr size_t FILEWRITER_CHUNK_SIZE = 10ull * 1024ull * 1024ull; // 10MB

/// <summary>
/// FileWriter provides endianess independent file write methods.
/// </summary>
class FileWriter : public virtual File
{
  public:
    enum class FileOpenMode // NOLINT
    {
        Normal,
        EraseOldFile,
        Append
    };

    struct Status
    {
        size_t bytesToWrite;
        size_t bytesWritten;
    };

  private:
    std::shared_ptr<std::ostream> mOStream;
    std::function<void(Status)> mWriteStatusCallback;

  protected:
    std::shared_ptr<std::ostream> &GetWriteStream()
    {
        return mOStream;
    }
    std::function<void(Status)> &GetWriteStatusCallback()
    {
        return mWriteStatusCallback;
    }

  public:
    /// <summary>
    /// Creates a new FileWriter instance
    /// </summary>
    FileWriter();

    ~FileWriter() override = default;

    FileWriter(const FileWriter &) = default;
    FileWriter(FileWriter &&)      = default;

    FileWriter &operator=(const FileWriter &) = default;

    FileWriter &operator=(FileWriter &&aOther) noexcept;

    /// <summary>
    /// Opens the file and writes the entire content.
    /// </summary>
    virtual void OpenAndWrite() = 0;
    /// <summary>
    /// Converts from internal data type to Em data type enum<para/>
    /// </summary>
    virtual void SetDataType(opp::image::PixelTypeEnum) = 0;
    /// <summary>
    /// Sets the pixel size given in nm and converts them if needed to the internal unit.
    /// </summary>
    virtual void SetPixelSize(double aPixelSizeInNM) = 0;

    void SetWriteStatusCallback(std::function<void(Status)> aCallback)
    {
        mWriteStatusCallback = std::move(aCallback);
    }
    void ResetWriteStatusCallback()
    {
        mWriteStatusCallback = nullptr;
    }

    /// <summary>
    /// Sets the filename to use for any write to disk operation
    /// </summary>
    void SetFileName(const std::filesystem::path &aFilename);

    /// <summary>
    /// Saves the content of the in-memory data to file aFileName and sets the internal filename
    /// </summary>
    virtual void SaveAs(const std::filesystem::path &aFileName) = 0;

  protected:
    template <typename T> void WriteLE(T &aValue)
    {
        if (!IsLittleEndian())
        {
            EndianSwap(aValue);
        }

        mOStream->write(reinterpret_cast<char *>(&aValue), sizeof(T));
    }

    template <typename T> void WriteBE(T &aValue)
    {
        if (IsLittleEndian())
        {
            EndianSwap(aValue);
        }

        mOStream->write(reinterpret_cast<char *>(&aValue), sizeof(T));
    }

    template <typename T> void WriteLE(std::vector<T> &aValues)
    {
        if (!IsLittleEndian())
        {
            EndianSwap(aValues.begin(), aValues.end());
        }

        mOStream->write(reinterpret_cast<char *>(&aValues.data()), sizeof(T) * aValues.size());
    }

    template <typename T> void WriteBE(std::vector<T> &aValues)
    {
        if (IsLittleEndian())
        {
            EndianSwap(aValues.begin(), aValues.end());
        }

        mOStream->write(reinterpret_cast<char *>(&aValues.data()), sizeof(T) * aValues.size());
    }

    void Write(sbyte &aValue);
    void Write(byte &aValue);

    /// <summary>
    /// Open the filestream for writing. Throws FileIOException if not successful.
    /// </summary>
    void OpenFileForWriting(FileOpenMode aMode);
    void CloseFileForWriting();

    void Write(const char *aSrc, size_t aCount);
    void WriteWithStatus(const char *aSrc, size_t aCount);

    void SeekWrite(size_t aPos, std::ios_base::seekdir aDir = std::ios_base::beg);
    size_t TellWrite();
};
} // namespace opp::fileIO