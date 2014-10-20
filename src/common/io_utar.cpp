#include "io_utar.h"
#include "logger.h"
#include <fstream>
#include <cmath>
#include <cstdint>
#include <boost/iostreams/device/file.hpp>
#include <boost/iostreams/filtering_stream.hpp>
#include <boost/iostreams/filter/gzip.hpp>
#include <boost/iostreams/filter/bzip2.hpp>
#include <boost/algorithm/string.hpp>

namespace ncv
{
        namespace
        {
                using std::int64_t;
                using std::uint64_t;

                #define ASCII_TO_NUMBER(num) ((num)-48) //Converts an ascii digit to the corresponding number (assuming it is an ASCII digit)

                /**
                 * Decode a TAR octal number.
                 * Ignores everything after the first NUL or space character.
                 * @param data A pointer to a size-byte-long octal-encoded
                 * @param size The size of the field pointer to by the data pointer
                 * @return
                 */
                static uint64_t decodeTarOctal(char* data, size_t size = 12)
                {
                        unsigned char* currentPtr = (unsigned char*) data + size;
                        uint64_t sum = 0;
                        uint64_t currentMultiplier = 1;
                        //Skip everything after the last NUL/space character
                        //In some TAR archives the size field has non-trailing NULs/spaces, so this is neccessary
                        unsigned char* checkPtr = currentPtr; //This is used to check where the last NUL/space char is
                        for (; checkPtr >= (unsigned char*) data; checkPtr--)
                        {
                                if ((*checkPtr) == 0 || (*checkPtr) == ' ')
                                {
                                        currentPtr = checkPtr - 1;
                                }
                        }
                        for (; currentPtr >= (unsigned char*) data; currentPtr--)
                        {
                                sum += ASCII_TO_NUMBER(*currentPtr) * currentMultiplier;
                                currentMultiplier *= 8;
                        }
                        return sum;
                }

                struct TARFileHeader
                {
                        char filename[100]; //NUL-terminated
                        char mode[8];
                        char uid[8];
                        char gid[8];
                        char fileSize[12];
                        char lastModification[12];
                        char checksum[8];
                        char typeFlag; //Also called link indicator for none-UStar format
                        char linkedFileName[100];
                        //USTar-specific fields -- NUL-filled in non-USTAR version
                        char ustarIndicator[6]; //"ustar" -- 6th character might be NUL but results show it doesn't have to
                        char ustarVersion[2]; //00
                        char ownerUserName[32];
                        char ownerGroupName[32];
                        char deviceMajorNumber[8];
                        char deviceMinorNumber[8];
                        char filenamePrefix[155];
                        char padding[12]; //Nothing of interest, but relevant for checksum

                        bool isUSTAR()
                        {
                                return (memcmp("ustar", ustarIndicator, 5) == 0);
                        }

                        size_t getFileSize()
                        {
                                return decodeTarOctal(fileSize);
                        }

                        bool checkChecksum()
                        {
                                //We need to set the checksum to zero
                                char originalChecksum[8];
                                memcpy(originalChecksum, checksum, 8);
                                memset(checksum, ' ', 8);
                                //Calculate the checksum -- both signed and unsigned
                                int64_t unsignedSum = 0;
                                int64_t signedSum = 0;
                                for (size_t i = 0; i < sizeof(TARFileHeader); i ++)
                                {
                                        unsignedSum += ((unsigned char*) this)[i];
                                        signedSum += ((signed char*) this)[i];
                                }
                                //Copy back the checksum
                                memcpy(checksum, originalChecksum, 8);
                                //Decode the original checksum
                                uint64_t referenceChecksum = decodeTarOctal(originalChecksum);
                                return  referenceChecksum == static_cast<uint64_t>(unsignedSum) ||
                                        referenceChecksum == static_cast<uint64_t>(signedSum);
                        }
                };
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
                if (boost::algorithm::iends_with(path, ".gz"))
                {
                        in.push(boost::iostreams::gzip_decompressor());
                }
                else if (boost::algorithm::iends_with(path, ".bz2"))
                {
                        in.push(boost::iostreams::bzip2_decompressor());
                }
                else if (boost::algorithm::iends_with(path, ".tar"))
                {
                        // no decompression filter needed
                }
                else
                {
                        log_error() << error_header << "unknown file suffix <" << path << ">!";
                        return false;
                }
                in.push(fin);

                // initialize a zero-filled block we can compare against (zero-filled header block --> end of TAR archive)
                char zeroBlock[512];
                memset(zeroBlock, 0, 512);

                // read file
                bool nextEntryHasLongName = false;
                while (in)
                {
                        TARFileHeader currentFileHeader;
                        in.read((char*)&currentFileHeader, 512);
                        //When a block with zeroes-only is found, the TAR archive ends here
                        if (memcmp(&currentFileHeader, zeroBlock, 512) == 0)
                        {
                                log_info() << info_header << "found TAR end";
                                break;
                        }

                        //Uncomment this to check all header checksums
                        //There seem to be TARs on the internet which include single headers that do not match the checksum even if most headers do.
                        //This might indicate a code error.
                        //assert(currentFileHeader.checkChecksum());

                        //Uncomment this to check for USTAR if you need USTAR features
                        //assert(currentFileHeader.isUSTAR());

                        //Convert the filename to a std::string to make handling easier
                        //Filenames of length 100+ need special handling
                        // (only USTAR supports 101+-character filenames, but in non-USTAR archives the prefix is 0 and therefore ignored)
                        std::string filename(currentFileHeader.filename, std::min((size_t)100, strlen(currentFileHeader.filename)));
                        //---Remove the next block if you don't want to support long filenames---
                        size_t prefixLength = strlen(currentFileHeader.filenamePrefix);
                        if (prefixLength > 0)
                        {
                                //If there is a filename prefix, add it to the string. See `man ustar`LON
                                filename = std::string(currentFileHeader.filenamePrefix, std::min((size_t)155, prefixLength)) + "/" + filename; //min limit: Not needed by spec, but we want to be safe
                        }

                        //Ignore directories, only handle normal files (symlinks are currently ignored completely and might cause errors)
                        if (currentFileHeader.typeFlag == '0' || currentFileHeader.typeFlag == 0)
                        {
                                //Normal file
                                //Handle GNU TAR long filenames -- the current block contains the filename only whilst the next block contains metadata
                                if (nextEntryHasLongName)
                                {
                                        //Set the filename from the current header
                                        filename = std::string(currentFileHeader.filename);
                                        //The next header contains the metadata, so replace the header before reading the metadata
                                        in.read((char*) &currentFileHeader, 512);
                                        //Reset the long name flag
                                        nextEntryHasLongName = false;
                                }

                                //Now the metadata in the current file header is valie -- we can read the values.
                                size_t size = currentFileHeader.getFileSize();
                                //Log that we found a file
                                log_info() << info_header << "found file <" << filename << "> (" << size << " bytes).";

                                //Read the file into memory
                                //  This won't work for very large files -- use streaming methods there!
                                std::vector<unsigned char> fileData(size + 1); //+1: Place a terminal NUL to allow interpreting the file as cstring (you can remove this if unused)
                                in.read(reinterpret_cast<char*>(fileData.data()), size);
                                callback(filename, fileData);

                                //In the tar archive, entire 512-byte-blocks are used for each file
                                //Therefore we now have to skip the padded bytes.
                                size_t paddingBytes = (512 - (size % 512)) % 512; //How long the padding to 512 bytes needs to be
                                //Simply ignore the padding
                                in.ignore(paddingBytes);
                                //----Remove the else if and else branches if you want to handle normal files only---
                        }
                        else if (currentFileHeader.typeFlag == '5')
                        {
                                //A directory
                                //Currently long directory names are not handled correctly
                                log_info() << info_header << "found directory <" << filename << ">.";
                        }
                        else if(currentFileHeader.typeFlag == 'L')
                        {
                                nextEntryHasLongName = true;
                        }
                        else
                        {
                                //Neither normal file nor directory (symlink etc.) -- currently ignored silently
                                log_info() << info_header << "found unhandled TAR Entry type <" << currentFileHeader.typeFlag << ">.";
                        }
                }

                // OK
                return true;
        }
}
