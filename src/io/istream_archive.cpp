#include "istream_archive.h"
#include <archive.h>
#include <archive_entry.h>

using namespace nano;

archive_istream_t::archive_istream_t(archive* ar) :
        m_archive(ar)
{
}

io_status archive_istream_t::advance(const std::streamsize num_bytes, buffer_t& buffer)
{
        while (static_cast<std::streamsize>(buffer.size()) < num_bytes)
        {
                const void* buff = nullptr;
                size_t size = 0;
                off_t offset;

                switch (archive_read_data_block(m_archive, &buff, &size, &offset))
                {
                case ARCHIVE_EOF:       return io_status::eof;
                case ARCHIVE_OK:        break;
                default:                return io_status::error;
                }

                buffer.insert(buffer.end(),
                        reinterpret_cast<const char*>(buff),
                        reinterpret_cast<const char*>(buff) + size);
        }

        return io_status::good;
}
