#include "archive.h"
#include "istream_archive.h"
#include <archive.h>
#include <archive_entry.h>

namespace nano
{
        static bool decode(archive* ar,
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

                        archive_istream_t stream(ar);
                        ok = callback(filename, stream);
                }

                archive_read_close(ar);
                archive_read_free(ar);

                // OK
                return ok;
        }

        bool load_archive(const std::string& path,
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
