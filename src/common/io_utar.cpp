#include "io_utar.h"
#include "logger.h"
#include <fstream>
#include <cmath>
#include <cstdint>
#include <boost/iostreams/device/file.hpp>
#include <boost/iostreams/filtering_stream.hpp>
#include <boost/iostreams/filter/gzip.hpp>
#include <boost/iostreams/filter/bzip2.hpp>
#include <boost/iostreams/stream.hpp>
#include <boost/iostreams/device/array.hpp>
#include <boost/algorithm/string.hpp>

namespace ncv
{
        // http://techoverflow.net/blog/2013/03/29/reading-tar-files-in-c/

        namespace
        {
                using std::int64_t;
                using std::uint64_t;

                ///
                /// \brief convert an ascii digit to the corresponding number (assuming it is an ASCII digit)
                ///
                uint64_t ascii_to_number(unsigned char num)
                {
                        return ((num) - 48);
                }

                ///
                /// \brief decode a TAR octal number. ignores everything after the first NUL or space character.
                ///
                uint64_t decode_tar_octal(const char* data, size_t size = 12)
                {
                        const unsigned char* currentPtr = (const unsigned char*)data + size;

                        const unsigned char* checkPtr = currentPtr;
                        for (; checkPtr >= (const unsigned char*) data; checkPtr --)
                        {
                                if ((*checkPtr) == 0 || (*checkPtr) == ' ')
                                {
                                        currentPtr = checkPtr - 1;
                                }
                        }

                        uint64_t sum = 0;
                        uint64_t currentMultiplier = 1;
                        for (; currentPtr >= (const unsigned char*)data; currentPtr --)
                        {
                                sum += ascii_to_number(*currentPtr) * currentMultiplier;
                                currentMultiplier *= 8;
                        }

                        return sum;
                }

                struct TARFileHeader
                {
                        char filename[100];
                        char mode[8];
                        char uid[8];
                        char gid[8];
                        char fileSize[12];
                        char lastModification[12];
                        char checksum[8];
                        char typeFlag;
                        char linkedFileName[100];
                        char ustarIndicator[6];
                        char ustarVersion[2];
                        char ownerUserName[32];
                        char ownerGroupName[32];
                        char deviceMajorNumber[8];
                        char deviceMinorNumber[8];
                        char filenamePrefix[155];
                        char padding[12];

                        size_t filesize() const
                        {
                                return decode_tar_octal(fileSize);
                        }
                };
        }

        namespace
        {
                bool io_untar(
                        boost::iostreams::filtering_istream& in, const io::untar_callback_t& callback,
                        const std::string& info_header, const std::string& error_header)
                {
                        char zeroBlock[512];
                        memset(zeroBlock, 0, 512);

                        bool nextEntryHasLongName = false;
                        while (in)
                        {
                                TARFileHeader header;
                                in.read((char*)&header, 512);
                                if (memcmp(&header, zeroBlock, 512) == 0)
                                {
                                        log_info() << info_header << "found TAR end.";
                                        break;
                                }

                                // compose the filename
                                std::string filename(header.filename, std::min((size_t)100, strlen(header.filename)));
                                const size_t prefixLength = strlen(header.filenamePrefix);
                                if (prefixLength > 0)
                                {
                                        filename =
                                        std::string(header.filenamePrefix, std::min((size_t)155, prefixLength)) +
                                        "/" +
                                        filename;
                                }

                                if (header.typeFlag == '0' || header.typeFlag == 0)
                                {
                                        // handle GNU TAR long filenames
                                        if (nextEntryHasLongName)
                                        {
                                                filename = std::string(header.filename);
                                                in.read((char*) &header, 512);
                                                nextEntryHasLongName = false;
                                        }

                                        const size_t size = header.filesize();
                                        log_info() << info_header << "found file <" << filename << "> (" << size << " bytes).";

                                        //Read the file into memory
                                        //  This won't work for very large files -- use streaming methods there!
                                        {
                                                std::vector<unsigned char> filedata(size);

                                                char* const pdata = reinterpret_cast<char*>(filedata.data());
                                                in.read(pdata, size);

                                                // decode archive type
                                                if (    boost::algorithm::iends_with(filename, ".tar.gz") ||
                                                        boost::algorithm::iends_with(filename, ".tgz"))
                                                {
                                                        boost::iostreams::filtering_istream in_;
                                                        in_.push(boost::iostreams::gzip_decompressor());
                                                        in_.push(boost::iostreams::basic_array_source<char>(pdata, filedata.size()));
                                                        if (!io_untar(in_, callback, info_header, error_header))
                                                        {
                                                                return false;
                                                        }
                                                }
                                                else if (boost::algorithm::iends_with(filename, ".tar.bz2") ||
                                                         boost::algorithm::iends_with(filename, ".tbz"))
                                                {
                                                        boost::iostreams::filtering_istream in_;
                                                        in_.push(boost::iostreams::bzip2_decompressor());
                                                        in_.push(boost::iostreams::basic_array_source<char>(pdata, filedata.size()));
                                                        if (!io_untar(in_, callback, info_header, error_header))
                                                        {
                                                                return false;
                                                        }
                                                }
                                                else if (boost::algorithm::iends_with(filename, ".tar"))
                                                {
                                                        // no decompression filter needed
                                                        boost::iostreams::filtering_istream in_;
                                                        in_.push(boost::iostreams::basic_array_source<char>(pdata, filedata.size()));
                                                        if (!io_untar(in_, callback, info_header, error_header))
                                                        {
                                                                return false;
                                                        }
                                                }
                                                else
                                                {
                                                        callback(filename, filedata);
                                                }
                                        }

                                        // ignore padding
                                        const size_t paddingBytes = (512 - (size % 512)) % 512;
                                        in.ignore(paddingBytes);
                                }

                                else if (header.typeFlag == '5')
                                {
                                        log_info() << info_header << "found directory <" << filename << ">.";
                                }

                                else if(header.typeFlag == 'L')
                                {
                                        nextEntryHasLongName = true;
                                }

                                else
                                {
                                        log_info() << info_header << "found unhandled TAR entry type <" << header.typeFlag << ">.";
                                }
                        }

                        // OK
                        return true;
                }
        }

        bool io::untar(
                const std::string& path, const untar_callback_t& callback,
                const std::string& info_header, const std::string& error_header)
        {
                std::ifstream fin(path.c_str(), std::ios_base::in | std::ios_base::binary);
                if (!fin.is_open())
                {
                        log_error() << error_header << "failed to open file <" << path << ">!";
                        return false;
                }

                boost::iostreams::filtering_istream in;

                // decode archive type
                if (    boost::algorithm::iends_with(path, ".tar.gz") ||
                        boost::algorithm::iends_with(path, ".tgz"))
                {
                        in.push(boost::iostreams::gzip_decompressor());
                }
                else if (boost::algorithm::iends_with(path, ".tar.bz2") ||
                         boost::algorithm::iends_with(path, ".tbz"))
                {
                        in.push(boost::iostreams::bzip2_decompressor());
                }
                else if (boost::algorithm::iends_with(path, ".tar"))
                {
                        // no decompression filter needed
                }
                else if (boost::algorithm::iends_with(path, ".gz"))
                {
                        in.push(boost::iostreams::gzip_decompressor());
                        in.push(fin);

                        std::vector<unsigned char> filedata;

                        const size_t chunk = 4096;
                        char data[chunk];

                        std::streamsize read_size;
                        while ((read_size = boost::iostreams::read(in, data, chunk)) > 0)
                        {
                                filedata.insert(filedata.end(),
                                                reinterpret_cast<const unsigned char*>(data),
                                                reinterpret_cast<const unsigned char*>(data) + read_size);
                        }

                        callback(path, filedata);
                        return true;
                }
                else
                {
                        log_error() << error_header << "unknown file suffix <" << path << ">!";
                        return false;
                }

                in.push(fin);

                return io_untar(in, callback, info_header, error_header);
        }
}
