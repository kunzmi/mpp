#pragma once

#include "common/defines.h"
#include "fileReader.h"
#include <common/image/pixelTypes.h>
#include <common/image/size2D.h>
#include <filesystem>
#include <functional>
#include <istream>
#include <memory>

namespace opp::fileIO
{
/// <summary>
/// PseudoFileReader implements FileReader to provide reading capabilities from a given stream.
/// </summary>
class PseudoFileReader : public FileReader
{
  protected:
    PseudoFileReader();

  public:
    /// <summary>
    /// Creates a new PseudoFileReader instance for file stream aStream.
    /// </summary>
    explicit PseudoFileReader(std::shared_ptr<std::istream> &aStream);

    ~PseudoFileReader() override = default;

    PseudoFileReader(const PseudoFileReader &) = default;
    PseudoFileReader(PseudoFileReader &&)      = default;

    PseudoFileReader &operator=(const PseudoFileReader &) = default;
    PseudoFileReader &operator=(PseudoFileReader &&)      = default;

    /// <summary>
    /// Opens the file and reads the entire content.
    /// </summary>
    void OpenAndRead() override;
    /// <summary>
    /// Opens the file and reads only the file header.
    /// </summary>
    void OpenAndReadHeader() override;
    /// <summary>
    /// Opens the file and reads only the file header (no exception: returns false on failure).
    /// </summary>
    [[nodiscard]] bool TryToOpenAndReadHeader() noexcept override;
    /// <summary>
    /// Converts from file data type enum to internal data type
    /// </summary>
    [[nodiscard]] opp::image::PixelTypeEnum GetDataType() const override;
    /// <summary>
    /// Returns the image dimensions stored in the file header.
    /// </summary>
    [[nodiscard]] Vector3<int> Size() const override;
    /// <summary>
    /// Returns the dimensions of an X/Y-image plane stored in the file header.
    /// </summary>
    [[nodiscard]] opp::image::Size2D SizePlane() const override;
    /// <summary>
    /// Returns the pixel size stored in file header converted to nm.
    /// </summary>
    [[nodiscard]] double PixelSize() const override;
    /// <summary>
    /// Returns the inner data pointer.
    /// </summary>
    [[nodiscard]] void *Data() override;
    /// <summary>
    /// Returns the inner data pointer shifted to image plane aIdx.
    /// </summary>
    [[nodiscard]] void *Data(size_t aIdx) override;
    /// <summary>
    /// Returns the size of the data block. If the header is not yet read, it will return 0.
    /// </summary>
    [[nodiscard]] size_t DataSize() const override;
    /// <summary>
    /// Returns the size of one 2D image slice in bytes. If the header is not yet read, it will return 0.
    /// </summary>
    [[nodiscard]] size_t GetImageSizeInBytes() const override;
    /// <summary>
    /// Reads a specific slice of a 3D data set (movie stack, tilt-series, volume) from file.
    /// </summary>
    void ReadSlice(size_t aIdx) override;

    /// <summary>
    /// Reads specific slices of a 3D data set (movie stack, tilt-series, volume) from file.
    /// </summary>
    void ReadSlices(size_t aStartIdx, size_t aSliceCount) override;

    /// <summary>
    /// Reads a specific slice of a 3D data set (movie stack, tilt-series, volume) from file.
    /// </summary>
    void ReadSlice(void *aData, size_t aIdx) override;

    /// <summary>
    /// Reads specific slices of a 3D data set (movie stack, tilt-series, volume) from file.
    /// </summary>
    void ReadSlices(void *aData, size_t aStartIdx, size_t aSliceCount) override;

    /// <summary>
    /// Reads raw data from file to user provided memory. Offset is from the start of data block without any header.
    /// </summary>
    void ReadRaw(void *aData, size_t aSizeInBytes, size_t aOffset = 0) override;

    /// <summary>
    /// Returns the file type: UNKNOWN
    /// </summary>
    [[nodiscard]] FileType GetFileType() const override;
};
} // namespace opp::fileIO