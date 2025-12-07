#pragma once

#include "dllexport_fileio.h"
#include "file.h"
#include <common/defines.h>
#include <common/image/pixelTypes.h>
#include <common/image/size2D.h>
#include <common/vector3.h>
#include <filesystem>
#include <functional>
#include <istream>
#include <memory>
#include <typeinfo>

namespace mpp::fileIO
{
constexpr size_t FILEREADER_CHUNK_SIZE = 10ull * 1024ull * 1024ull; // 10MB

/// <summary>
/// FileReader provides endianess independent file read methods.
/// </summary>
class MPPEXPORT_COMMON_FILEIO FileReader : public virtual File
{
  public:
    struct Status
    {
        size_t bytesToRead;
        size_t bytesRead;
    };

  private:
    bool mIsStreamOwner{true};
    std::shared_ptr<std::istream> mIStream;
    std::function<void(Status)> mReadStatusCallback;

  protected:
    std::shared_ptr<std::istream> &ReadStream()
    {
        return mIStream;
    }

    std::function<void(Status)> &ReadStatusCallback()
    {
        return mReadStatusCallback;
    }

  public:
    /// <summary>
    /// Creates a new FileReader instance.
    /// </summary>
    FileReader();

    /// <summary>
    /// Creates a new FileReader instance for file stream aStream.
    /// </summary>
    explicit FileReader(std::shared_ptr<std::istream> &aStream);

    ~FileReader() override = default;

    FileReader(const FileReader &) = default;
    FileReader(FileReader &&)      = default;

    FileReader &operator=(const FileReader &) = default;

    FileReader &operator=(FileReader &&aOther) noexcept;

    /// <summary>
    /// Opens the file and reads the entire content.
    /// </summary>
    virtual void OpenAndRead() = 0;
    /// <summary>
    /// Opens the file and reads only the file header.
    /// </summary>
    virtual void OpenAndReadHeader() = 0;
    /// <summary>
    /// Opens the file and reads only the file header (no exception: returns false on failure).
    /// </summary>
    [[nodiscard]] virtual bool TryToOpenAndReadHeader() noexcept = 0;
    /// <summary>
    /// Converts from file data type enum to internal data type
    /// </summary>
    [[nodiscard]] virtual mpp::image::PixelTypeEnum GetDataType() const = 0;
    /// <summary>
    /// Returns the image dimensions stored in the file header.
    /// </summary>
    [[nodiscard]] virtual Vector3<int> Size() const = 0;
    /// <summary>
    /// Returns the dimensions of an X/Y-image plane stored in the file header.
    /// </summary>
    [[nodiscard]] virtual mpp::image::Size2D SizePlane() const = 0;
    /// <summary>
    /// Returns the pixel size stored in file header converted to nm.
    /// </summary>
    [[nodiscard]] virtual double PixelSize() const = 0;
    /// <summary>
    /// Returns the inner data pointer.
    /// </summary>
    [[nodiscard]] virtual void *Data() = 0;
    /// <summary>
    /// Returns the inner data pointer shifted to image plane aIdx.
    /// </summary>
    [[nodiscard]] virtual void *Data(size_t aIdx) = 0;
    /// <summary>
    /// Returns the size of the data block. If the header is not yet read, it will return 0.
    /// </summary>
    [[nodiscard]] virtual size_t DataSize() const = 0;
    /// <summary>
    /// Returns the size of one 2D image slice in bytes. If the header is not yet read, it will return 0.
    /// </summary>
    [[nodiscard]] virtual size_t GetImageSizeInBytes() const = 0;
    /// <summary>
    /// Reads a specific slice of a 3D data set (movie stack, tilt-series, volume) from file.
    /// </summary>
    virtual void ReadSlice(size_t aIdx) = 0;
    /// <summary>
    /// Reads specific slices of a 3D data set (movie stack, tilt-series, volume) from file.
    /// </summary>
    virtual void ReadSlices(size_t aStartIdx, size_t aSliceCount) = 0;
    /// <summary>
    /// Reads a specific slice of a 3D data set (movie stack, tilt-series, volume) from file.
    /// </summary>
    virtual void ReadSlice(void *aData, size_t aIdx) = 0;
    /// <summary>
    /// Reads specific slices of a 3D data set (movie stack, tilt-series, volume) from file.
    /// </summary>
    virtual void ReadSlices(void *aData, size_t aStartIdx, size_t aSliceCount) = 0;
    /// <summary>
    /// Reads raw data from file to user provided memory. Offset is from the start of data block without any header.
    /// </summary>
    virtual void ReadRaw(void *aData, size_t aSizeInBytes, size_t aOffset) = 0;

    /// <summary>
    /// When reading the file from disc, the callback function is called internally to tell the reading progress. To set
    /// before reading and to be reset afterwards.
    /// </summary>
    void SetReadStatusCallback(std::function<void(Status)> aCallback)
    {
        mReadStatusCallback = std::move(aCallback);
    }

    /// <summary>
    /// Reset the reading status callback
    /// </summary>
    void ResetReadStatusCallback()
    {
        mReadStatusCallback = nullptr;
    }

    /// <summary>
    /// Open the filestream for reading. Throws FileIOException if not successful
    /// </summary>
    void OpenFileForReading();

    /// <summary>
    /// Open the filestream for reading. Returns false on failure
    /// </summary>
    bool TryToOpenFileForReading() noexcept;

    /// <summary>
    /// Closes the filestream.
    /// </summary>
    void CloseFileForReading();

    /// <summary>
    /// Provides typesafe access to pixel data (first pixel of the first image). If the requested datatype doesn't match
    /// the file data, an exception is thrown.
    /// </summary>
    template <typename T> [[nodiscard]] T *Pixels()
    {

        if (mpp::image::pixel_type_enum<T>::pixelType != GetDataType())
        {
            throw FILEIOEXCEPTION(FileName(), "Trying to access pixel data with incompatible data type. Requested: "
                                                  << typeid(T).name() << " but pixel data type is: " << GetDataType());
        }
        return reinterpret_cast<T *>(Data());
    }

    /// <summary>
    /// Provides typesafe access to pixel data (first pixel of the image with index aIdx). If the requested datatype
    /// doesn't match the file data, an exception is thrown.
    /// </summary>
    template <typename T> [[nodiscard]] T *Pixels(size_t aIdx)
    {
        if (mpp::image::pixel_type_enum<T>::pixelType != GetDataType())
        {
            throw FILEIOEXCEPTION(FileName(), "Trying to access pixel data with incompatible data type. Requested: "
                                                  << typeid(T).name() << " but pixel data type is: " << GetDataType());
        }
        return reinterpret_cast<T *>(Data(aIdx));
    }

  protected:
    /// <summary>
    /// Creates a new FileReader instance for a dummy file.
    /// </summary>
    explicit FileReader(bool aIsDummy);

    template <typename T> T ReadLE()
    {
        T temp = T(0); // NOLINT(clang-analyzer-optin.core.EnumCastOutOfRange)
        mIStream->read(reinterpret_cast<char *>(&temp), sizeof(T));

        if (!IsLittleEndian())
        {
            EndianSwap(temp);
        }

        return temp;
    }

    template <typename T> T ReadBE()
    {
        T temp = T(0);
        mIStream->read(reinterpret_cast<char *>(&temp), sizeof(T));

        if (IsLittleEndian())
        {
            EndianSwap(temp);
        }

        return temp;
    }

    template <typename T> std::vector<T> ReadLE(size_t aCount)
    {
        std::vector<T> temp(aCount);
        mIStream->read(reinterpret_cast<char *>(temp.data()), std::streamsize(sizeof(T) * aCount));

        if (!IsLittleEndian())
        {
            EndianSwap(temp.begin(), temp.end());
        }

        return temp;
    }

    template <typename T> std::vector<T> ReadBE(size_t aCount)
    {
        std::vector<T> temp(aCount);
        mIStream->read(reinterpret_cast<char *>(temp.data()), std::streamsize(sizeof(T) * aCount));

        if (IsLittleEndian())
        {
            EndianSwap(temp.begin(), temp.end());
        }

        return temp;
    }

    sbyte ReadI1();
    byte ReadUI1();
    std::string ReadString(size_t aCount);

    void Read(char *aDest, size_t aCount);
    void ReadWithStatus(char *aDest, size_t aCount);

    void SeekRead(size_t aPos, std::ios_base::seekdir aDir = std::ios_base::beg);
    size_t TellRead();
};
} // namespace mpp::fileIO