#include "fileWriter.h"
#include <common/defines.h>
#include <common/fileIO/file.h>
#include <common/fileIO/fileIOException.h>
#include <common/safeCast.h>
#include <cstddef>
#include <filesystem>
#include <fstream>
#include <ios>
#include <memory>
#include <utility>

namespace opp::fileIO
{
FileWriter::FileWriter() : mOStream(nullptr)
{
}

FileWriter &FileWriter::operator=(FileWriter &&aOther) noexcept
{
    mOStream             = std::move(aOther.mOStream);
    mWriteStatusCallback = std::move(aOther.mWriteStatusCallback);
    return *this;
}

void FileWriter::SetFileName(const std::filesystem::path &aFilename)
{
    File::SetFileName(aFilename);
}

void FileWriter::OpenFileForWriting(FileOpenMode aMode)
{
    std::shared_ptr<std::ofstream> ofstr;

    switch (aMode)
    {
        case FileWriter::FileOpenMode::Normal:
            ofstr = std::make_shared<std::ofstream>(FileName().c_str(),
                                                    std::ios_base::in | std::ios_base::out | // NOLINT
                                                        std::ios_base::binary);              // NOLINT
            break;
        case FileWriter::FileOpenMode::EraseOldFile:
            ofstr = std::make_shared<std::ofstream>(FileName().c_str(),
                                                    std::ios_base::in | std::ios_base::out |           // NOLINT
                                                        std::ios_base::binary | std::ios_base::trunc); // NOLINT
            break;
        case FileWriter::FileOpenMode::Append:
            ofstr = std::make_shared<std::ofstream>(FileName().c_str(),
                                                    std::ios_base::in | std::ios_base::out |         // NOLINT
                                                        std::ios_base::binary | std::ios_base::app); // NOLINT
            break;
    }

    mOStream = ofstr;

    const bool ok = ofstr->is_open() && ofstr->good();
    if (!ok)
    {
        throw FILEIOEXCEPTION(FileName(), "Cannot open file for writing.");
    }
}

void FileWriter::CloseFileForWriting()
{
    const std::shared_ptr<std::ofstream> ofstr = std::dynamic_pointer_cast<std::ofstream>(mOStream);
    ofstr->close();
}

void FileWriter::Write(byte &aValue)
{
    mOStream->write(reinterpret_cast<char *>(&aValue), 1);
}

void FileWriter::Write(sbyte &aValue)
{
    mOStream->write(reinterpret_cast<char *>(&aValue), 1);
}

void FileWriter::SeekWrite(size_t aPos, std::ios_base::seekdir aDir)
{
    mOStream->seekp(std::streamoff(aPos), aDir);
}

size_t FileWriter::TellWrite()
{
    return to_size_t(mOStream->tellp());
}

void FileWriter::Write(const char *aSrc, size_t aCount)
{
    mOStream->write(aSrc, std::streamoff(aCount));
}

void FileWriter::WriteWithStatus(const char *aSrc, size_t aCount)
{
    if (mWriteStatusCallback)
    {
        const Status status{aCount, 0};
        mWriteStatusCallback(status);
    }

    if (aCount <= FILEWRITER_CHUNK_SIZE)
    {
        mOStream->write(aSrc, std::streamoff(aCount));
        if (mWriteStatusCallback)
        {
            const Status status{aCount, aCount};
            mWriteStatusCallback(status);
        }
    }
    else
    {
        for (size_t sizeWrite = 0; sizeWrite < aCount; sizeWrite += FILEWRITER_CHUNK_SIZE)
        {
            size_t sizeToWrite = FILEWRITER_CHUNK_SIZE;
            if (sizeWrite + sizeToWrite > aCount)
            {
                sizeToWrite = aCount - sizeWrite;
            }

            mOStream->write(aSrc + sizeWrite, std::streamoff(sizeToWrite));

            if (mWriteStatusCallback)
            {
                const Status status{aCount, sizeWrite + sizeToWrite};
                mWriteStatusCallback(status);
            }
        }
    }
}
} // namespace opp::fileIO