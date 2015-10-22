#include "archive.h"
#include "text/ends_with.hpp"
#include <istream>
#include <archive.h>
#include <archive_entry.h>

#include "cortex/logger.h"

namespace cortex
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

        static bool copy(archive* ar, buffer_t& data)
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

        bool decode(const buffer_t& buffer,
                const archive_callback_t&,
                const archive_error_callback_t&);

        bool decode(archive* ar,
                const archive_callback_t& callback,
                const archive_error_callback_t& error_callback)
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
                                error_callback("failed to read archive!");
                                error_callback(std::string("error <") + archive_error_string(ar) + ">!");
                                ok = false;
                                break;
                        }

                        const std::string filename = archive_entry_pathname(entry);
                        const archive_type filetype = decode_archive_type(filename);
//                        const int64_t filesize = archive_entry_size(entry);

                        buffer_t data;
                        if (!copy(ar, data))
                        {
                                error_callback("failed to read archive!");
                                error_callback(std::string("error <") + archive_error_string(ar) + ">!");
                                ok = false;
                                break;
                        }

                        log_info() << "decode: filename = " << filename << ", data.size() = " << data.size();

                        switch (filetype)
                        {
                        case archive_type::tar:
                        case archive_type::tar_gz:
                        case archive_type::tar_bz2:
                        case archive_type::gz:
                        case archive_type::bz2:
                                ok = decode(data, callback, error_callback);
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

        bool decode(const buffer_t& buffer,
                const archive_callback_t& callback,
                const archive_error_callback_t& error_callback)
        {
                archive* ar = archive_read_new();

                archive_read_support_filter_all(ar);
                archive_read_support_format_all(ar);
                archive_read_support_format_raw(ar);

                if (archive_read_open_memory(ar, (void*)buffer.data(), buffer.size()))
                {
                        error_callback("failed to open archive!");
                        error_callback(std::string("error <") + archive_error_string(ar) + ">!");
                        return false;
                }

                return decode(ar, callback, error_callback);
        }

        bool unarchive(const std::string& path,
                const archive_callback_t& callback,
                const archive_error_callback_t& error_callback)
        {
                archive* ar = archive_read_new();

                archive_read_support_filter_all(ar);
                archive_read_support_format_all(ar);
                archive_read_support_format_raw(ar);

                if (archive_read_open_filename(ar, path.c_str(), 10240))
                {
                        error_callback("failed to open archive <" + path + ">!");
                        error_callback(std::string("error <") + archive_error_string(ar) + ">!");
                        return false;
                }

                return decode(ar, callback, error_callback);
        }
}
