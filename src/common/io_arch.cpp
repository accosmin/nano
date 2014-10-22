#include "io_arch.h"
#include "io_bzip.h"
#include "io_gzip.h"
#include "logger.h"
#include <fstream>
#include <cmath>
#include <cstdint>
#include <archive.h>
#include <archive_entry.h>

namespace ncv
{
//        static bool io_decode(const io::data_t& orig_data, const std::string& path, io::data_t& data)
//        {
//                const archive_type type = decode_archive_type(path);

//                switch (type)
//                {
//                case archive_type::tar_gz:
//                case archive_type::gz:
//                        return io::uncompress_gzip(orig_data, data);

//                case archive_type::tar_bz2:
//                case archive_type::bz2:
//                        return io::uncompress_bzip2(orig_data, data);

//                case archive_type::tar:
//                default:
//                        return !(data = orig_data).empty();
//                }
//        }

//        static bool io_decode(std::istream& istream, const std::string& path, io::data_t& data)
//        {
//                const archive_type type = decode_archive_type(path);

//                switch (type)
//                {
//                case archive_type::tar_gz:
//                case archive_type::gz:
//                        return io::uncompress_gzip(istream, data);

//                case archive_type::tar_bz2:
//                case archive_type::bz2:
//                        return io::uncompress_bzip2(istream, data);

//                case archive_type::tar:
//                default:
//                        return io::load_binary(istream, data);
//                }
//        }

//        static bool io_untar(const io::data_t& data, const std::string& log_header, const io::data_callback_t& callback)
//        {
//                char zeroBlock[512];
//                memset(zeroBlock, 0, 512);

//                bool nextEntryHasLongName = false;

//                for (size_t pos = 0; pos < data.size(); )
//                {
//                        TARFileHeader header;
//                        if (!io::load_struct(data, header, pos))
//                        {
//                                log_error() << log_header << "failed to read TAR header!";
//                                return false;
//                        }
//                        if (memcmp(&header, zeroBlock, 512) == 0)
//                        {
//                                log_info() << log_header << "found TAR end.";
//                                break;
//                        }

//                        // compose the filename
//                        std::string filename(header.filename, std::min((size_t)100, strlen(header.filename)));
//                        const size_t prefixLength = strlen(header.filenamePrefix);
//                        if (prefixLength > 0)
//                        {
//                                filename =
//                                std::string(header.filenamePrefix, std::min((size_t)155, prefixLength)) +
//                                "/" +
//                                filename;
//                        }

//                        if (header.typeFlag == '0' || header.typeFlag == 0)
//                        {
//                                // handle GNU TAR long filenames
//                                if (nextEntryHasLongName)
//                                {
//                                        filename = std::string(header.filename);
//                                        if (!io::load_struct(data, header, pos))
//                                        {
//                                                log_error() << log_header << "failed to read TAR header!";
//                                                return false;
//                                        }
//                                        nextEntryHasLongName = false;
//                                }

//                                const size_t filesize = header.filesize();
//                                log_info() << log_header << "found file <" << filename << "> (" << filesize << " bytes).";

//                                // read the file into memory
//                                io::data_t orig_filedata;
//                                if (!io::load_data(data, filesize, orig_filedata, pos))
//                                {
//                                        log_error() << log_header << "failed to read TAR data!";
//                                        return false;
//                                }

//                                io::data_t filedata;
//                                if (!io_decode(orig_filedata, filename, filedata))
//                                {
//                                        log_error() << log_header << "failed to decode TAR data!";
//                                        return false;
//                                }

//                                callback(filename, filedata);

//                                // ignore padding
//                                const size_t paddingBytes = (512 - (filesize % 512)) % 512;
//                                if (!io::load_skip(data, paddingBytes, pos))
//                                {
//                                        log_error() << log_header << "failed to skip TAR padding!";
//                                        return false;
//                                }
//                        }

//                        else if (header.typeFlag == '5')
//                        {
//                                log_info() << log_header << "found directory <" << filename << ">.";
//                        }

//                        else if(header.typeFlag == 'L')
//                        {
//                                nextEntryHasLongName = true;
//                        }

//                        else
//                        {
//                                log_info() << log_header << "found unhandled TAR entry type <" << header.typeFlag << ">.";
//                        }
//                }

//                // OK
//                return true;
//        }

        bool io::decode(const std::string& path, const std::string& log_header, const data_callback_t& callback)
        {
                int r;

                archive* a = archive_read_new();
//                archive_read_support_format_tar(a);
                archive_read_support_format_all(a);

                if ((r = archive_read_open_filename(a, path.c_str(), 10240)))
                {
                        log_error() << log_header << "failed to open archive <" << path << ">!";
                        log_error() << log_header << "error <" << archive_error_string(a) << ">!";
                        return false;
                }

                bool ok = true;
                while (true)
                {
                        archive_entry* entry;
                        r = archive_read_next_header(a, &entry);

                        if (r == ARCHIVE_EOF)
                                break;
                        if (r != ARCHIVE_OK)
                        {
                                log_warning() << log_header << "failed to read archive!";
                                log_warning() << log_header << "error <" << archive_error_string(a) << ">!";
                                ok = false;
                                break;
                        }

                        const std::string filename = archive_entry_pathname(entry);
                        log_info() << log_header << "filename = <" << filename << ">";

//                        copy_data(a, ext);
                }

                archive_read_close(a);
                archive_read_free(a);

                // OK
                return ok;
        }
}
