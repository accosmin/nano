#include "io_arch.h"
#include "io_bzip.h"
#include "io_gzip.h"
#include "logger.h"
#include <archive.h>
#include <archive_entry.h>
#include <boost/algorithm/string.hpp>

namespace ncv
{
        enum class archive_type : int
        {
                tar,
                tar_gz,
                tar_bz2,
                gz,
                bz2,
                unknown
        };

        static archive_type decode_archive_type(const std::string& path)
        {

                if (    boost::algorithm::iends_with(path, ".tar.gz") ||
                        boost::algorithm::iends_with(path, ".tgz"))
                {
                        return archive_type::tar_gz;
                }

                else if (boost::algorithm::iends_with(path, ".tar.bz2") ||
                         boost::algorithm::iends_with(path, ".tbz") ||
                         boost::algorithm::iends_with(path, ".tbz2") ||
                         boost::algorithm::iends_with(path, ".tb2"))
                {
                        return archive_type::tar_bz2;
                }

                else if (boost::algorithm::iends_with(path, ".tar"))
                {
                        return archive_type::tar;
                }

                else if (boost::algorithm::iends_with(path, ".gz"))
                {
                        return archive_type::gz;
                }

                else
                {
                        return archive_type::unknown;
                }
        }

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

        bool copy(archive* ar, io::data_t& data)
        {
                while (true)
                {
                        const void* buff;
                        size_t size;
                        off_t offset;

                        const int r = archive_read_data_block(ar, &buff, &size, &offset);
                        if (r == ARCHIVE_EOF)
                                return true;
                        if (r != ARCHIVE_OK)
                                return false;

                        data.insert(data.end(), (const char*)buff, (const char*)buff + size);
                }

                return true;
        }

        bool io::decode(const std::string& path, const std::string& log_header, const data_callback_t& callback)
        {
                int r;

                archive* ar = archive_read_new();

                archive_read_support_filter_all(ar);
                archive_read_support_format_all(ar);
                archive_read_support_format_raw(ar);

                if ((r = archive_read_open_filename(ar, path.c_str(), 10240)))
                {
                        log_error() << log_header << "failed to open archive <" << path << ">!";
                        log_error() << log_header << "error <" << archive_error_string(ar) << ">!";
                        return false;
                }

                bool ok = true;
                while (ok)
                {
                        archive_entry* entry;
                        r = archive_read_next_header(ar, &entry);

                        if (r == ARCHIVE_EOF)
                                break;
                        if (r != ARCHIVE_OK)
                        {
                                log_error() << log_header << "failed to read archive!";
                                log_error() << log_header << "error <" << archive_error_string(ar) << ">!";
                                ok = false;
                                break;
                        }

                        const std::string filename = archive_entry_pathname(entry);
                        const archive_type filetype = decode_archive_type(filename);
                        const int64_t filesize = archive_entry_size(entry);

                        log_info() << log_header << "filename = <" << filename << ">, filesize = <" << filesize << ">";

                        switch (filetype)
                        {
                        case archive_type::tar:
                        case archive_type::tar_gz:
                        case archive_type::tar_bz2:
                        case archive_type::gz:
                        case archive_type::bz2:
                                // todo
                                break;

                        default:
                                {
                                        data_t data;
                                        if (copy(ar, data))
                                        {
                                                log_info() << log_header
                                                           << "filename = <" << filename
                                                           << ">, filesize = <" << data.size()
                                                           << "/" << filesize << ">";
                                                callback(filename, data);
                                        }
                                        else
                                        {
                                                log_error() << log_header << "failed to read entry!";
                                                log_error() << log_header << "error <" << archive_error_string(ar) << ">!";
                                                ok = false;
                                        }
                                }
                                break;
                        }
                }

                archive_read_close(ar);
                archive_read_free(ar);

                // OK
                return ok;
        }
}
