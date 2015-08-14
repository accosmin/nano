#include "archive.h"
#include "bzip.h"
#include "gzip.h"
#include "nanocv/logger.h"
#include "nanocv/text.hpp"
#include <archive.h>
#include <archive_entry.h>

namespace ncv
{
        namespace detail
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

                archive_type decode_archive_type(const std::string& path)
                {
                        if (    text::iends_with(path, ".tar.gz") ||
                                text::iends_with(path, ".tgz"))
                        {
                                return archive_type::tar_gz;
                        }

                        else if (text::iends_with(path, ".tar.bz2") ||
                                 text::iends_with(path, ".tbz") ||
                                 text::iends_with(path, ".tbz2") ||
                                 text::iends_with(path, ".tb2"))
                        {
                                return archive_type::tar_bz2;
                        }

                        else if (text::iends_with(path, ".tar"))
                        {
                                return archive_type::tar;
                        }

                        else if (text::iends_with(path, ".gz"))
                        {
                                return archive_type::gz;
                        }

                        else
                        {
                                return archive_type::unknown;
                        }
                }

                bool copy(archive* ar, io::buffer_t& data)
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

                bool decode(const io::buffer_t& mem_data, const std::string& log_header, const io::buffer_callback_t& callback);

                bool decode(archive* ar, const std::string& log_header, const io::buffer_callback_t& callback)
                {
                        bool ok = true;
                        while (ok)
                        {
                                archive_entry* entry;
                                const int r = archive_read_next_header(ar, &entry);

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
                                const detail::archive_type filetype = detail::decode_archive_type(filename);
//                                const int64_t filesize = archive_entry_size(entry);

                                io::buffer_t data;
                                if (!detail::copy(ar, data))
                                {
                                        log_error() << log_header << "failed to read archive!";
                                        log_error() << log_header << "error <" << archive_error_string(ar) << ">!";
                                        ok = false;
                                        break;
                                }

                                switch (filetype)
                                {
                                case detail::archive_type::tar:
                                case detail::archive_type::tar_gz:
                                case detail::archive_type::tar_bz2:
                                case detail::archive_type::gz:
                                case detail::archive_type::bz2:
                                        ok = detail::decode(data, log_header, callback);
                                        break;

                                default:
                                        ok = callback(filename, data);
                                        break;
                                }
                        }

                        archive_read_close(ar);
                        archive_read_free(ar);

                        // OK
                        return ok;
                }

                bool decode(const io::buffer_t& mem_data, const std::string& log_header, const io::buffer_callback_t& callback)
                {
                        archive* ar = archive_read_new();

                        archive_read_support_filter_all(ar);
                        archive_read_support_format_all(ar);
                        archive_read_support_format_raw(ar);

                        if (archive_read_open_memory(ar, (void*)mem_data.data(), mem_data.size()))
                        {
                                log_error() << log_header << "failed to open archive!";
                                log_error() << log_header << "error <" << archive_error_string(ar) << ">!";
                                return false;
                        }

                        return decode(ar, log_header, callback);
                }
        }

        bool io::decode(const std::string& path, const std::string& log_header, const buffer_callback_t& callback)
        {
                archive* ar = archive_read_new();

                archive_read_support_filter_all(ar);
                archive_read_support_format_all(ar);
                archive_read_support_format_raw(ar);

                if (archive_read_open_filename(ar, path.c_str(), 10240))
                {
                        log_error() << log_header << "failed to open archive <" << path << ">!";
                        log_error() << log_header << "error <" << archive_error_string(ar) << ">!";
                        return false;
                }

                return detail::decode(ar, log_header, callback);
        }
}
