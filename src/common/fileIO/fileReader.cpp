#include "fileReader.h"
#include <common/defines.h>
#include <common/fileIO/fileIOException.h>
#include <common/safeCast.h>
#include <cstddef>
#include <exception>
#include <filesystem>
#include <fstream>
#include <ios>
#include <istream>
#include <memory>
#include <string>
#include <utility>
#include <vector>

namespace mpp::fileIO
{
FileReader::FileReader() : mIStream(nullptr)
{
}

FileReader::FileReader(std::shared_ptr<std::istream> &aStream) : mIsStreamOwner(false), mIStream(aStream)
{
}

FileReader::FileReader(bool aIsDummy) : mIsStreamOwner(!aIsDummy), mIStream(nullptr)
{
}

FileReader &FileReader::operator=(FileReader &&aOther) noexcept
{
    mIsStreamOwner      = aOther.mIsStreamOwner;
    mIStream            = std::move(aOther.mIStream);
    mReadStatusCallback = std::move(aOther.mReadStatusCallback);
    return *this;
}

void FileReader::OpenFileForReading()
{
    if (!mIsStreamOwner)
    {
        return;
    }
    const std::shared_ptr<std::ifstream> ifstr =
        std::make_shared<std::ifstream>(FileName().c_str(), std::ios_base::in | std::ios_base::binary); // NOLINT

    mIStream      = ifstr;
    const bool ok = ifstr->is_open() && ifstr->good();
    if (!ok)
    {
        throw FILEIOEXCEPTION(FileName(), "Cannot open file for reading.");
    }
}

bool FileReader::TryToOpenFileForReading() noexcept
{
    try
    {
        if (!mIsStreamOwner)
        {
            return false;
        }
        const std::shared_ptr<std::ifstream> ifstr =
            std::make_shared<std::ifstream>(FileName().c_str(), std::ios_base::in | std::ios_base::binary); // NOLINT

        mIStream      = ifstr;
        const bool ok = ifstr->is_open() && ifstr->good();
        return ok;
    }
    catch (const std::exception &)
    {
        return false;
    }
}

void FileReader::CloseFileForReading()
{
    if (!mIsStreamOwner)
    {
        return;
    }

    std::shared_ptr<std::ifstream> ifstr = std::dynamic_pointer_cast<std::ifstream>(mIStream);
    mIStream.reset();
    if (ifstr == nullptr)
    {
        return;
    }
    ifstr->close();
    ifstr.reset();
}

sbyte FileReader::ReadI1()
{
    char temp = 0;
    mIStream->read(&temp, 1);

    return temp;
}

byte FileReader::ReadUI1()
{
    byte temp = 0;
    mIStream->read(reinterpret_cast<char *>(&temp), 1);

    return temp;
}

std::string FileReader::ReadString(size_t aCount)
{
    if (aCount == 0)
    {
        return {};
    }

    std::vector<char> nameTemp(aCount + 1);
    nameTemp[aCount] = '\0';

    mIStream->read(nameTemp.data(), std::streamsize(aCount));

    std::string ret(nameTemp.data());
    return ret;
}

void FileReader::Read(char *aDest, size_t aCount)
{
    mIStream->read(aDest, std::streamsize(aCount));
}

void FileReader::ReadWithStatus(char *aDest, size_t aCount)
{
    if (mReadStatusCallback)
    {
        const Status status{aCount, 0};
        mReadStatusCallback(status);
    }

    if (aCount <= FILEREADER_CHUNK_SIZE)
    {
        mIStream->read(aDest, std::streamsize(aCount));

        if (mReadStatusCallback)
        {
            const Status status{aCount, aCount};
            mReadStatusCallback(status);
        }
    }
    else
    {
        for (size_t sizeRead = 0; sizeRead < aCount; sizeRead += FILEREADER_CHUNK_SIZE)
        {
            size_t sizeToRead = FILEREADER_CHUNK_SIZE;
            if (sizeRead + sizeToRead > aCount)
            {
                sizeToRead = aCount - sizeRead;
            }

            mIStream->read(aDest + sizeRead, std::streamsize(sizeToRead));

            if (mReadStatusCallback)
            {
                const Status status{aCount, sizeRead + sizeToRead};
                mReadStatusCallback(status);
            }
        }
    }
}

void FileReader::SeekRead(size_t aPos, std::ios_base::seekdir aDir)
{
    mIStream->seekg(std::streamoff(aPos), aDir);
}

size_t FileReader::TellRead()
{
    const size_t ret = to_size_t(mIStream->tellg());
    return ret;
}
} // namespace mpp::fileIO