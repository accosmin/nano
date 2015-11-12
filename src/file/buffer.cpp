#include "buffer.h"
#include "mstream.h"
#include <limits>
#include <fstream>


#include <iostream>

namespace file
{
        template
        <
                typename tstream,
                typename tsize
        >
        bool impl_load_buffer_from_stream(tstream& istream, tsize orig_num_bytes, buffer_t& buffer)
        {
                buffer.clear();

                static const std::streamsize chunk_size = 64 * 1024;
                char chunk[chunk_size];

                // read in  chunks
                std::streamsize num_bytes = static_cast<std::streamsize>(orig_num_bytes);
                while (num_bytes > 0 && istream)
                {
                        const std::streamsize to_read = (num_bytes >= chunk_size) ? chunk_size : num_bytes;
                        num_bytes -= to_read;

                        istream.read(chunk, to_read);
                        if (istream.good() || istream.eof())
                        {
                                buffer.insert(buffer.end(), chunk, chunk + istream.gcount());
                        }
                }

                // OK
                return (orig_num_bytes == max_streamsize()) ? istream.eof() : (num_bytes == 0);
        }

        bool load_buffer_from_stream(std::istream& istream, std::streamsize num_bytes, buffer_t& buffer)
        {
                return impl_load_buffer_from_stream(istream, num_bytes, buffer);
        }

        bool load_buffer_from_stream(std::istream& istream, buffer_t& buffer)
        {
                return impl_load_buffer_from_stream(istream, max_streamsize(), buffer);
        }

        bool load_buffer_from_stream(mstream_t& istream, buffer_t& buffer)
        {
                return impl_load_buffer_from_stream(istream, istream.size(), buffer);
        }

        std::streamsize max_streamsize()
        {
                return std::numeric_limits<std::streamsize>::max();
        }

        bool save_buffer(const std::string& path, const buffer_t& buffer)
        {
                std::ofstream stream(path.c_str(), std::ios::binary | std::ios::out);
                if (!stream.is_open())
                {
                        return false;
                }

                const std::streamsize stream_size = 64 * 1024;
                std::streamsize num_bytes = static_cast<std::streamsize>(buffer.size());

                const char* pbuffer = buffer.data();
                while (stream && num_bytes > 0)
                {
                        const auto to_write = num_bytes >= stream_size ? stream_size : num_bytes;

                        if (stream.write(pbuffer, to_write))
                        {
                                pbuffer += to_write;
                                num_bytes -= to_write;
                        }
                }

                return true;
        }

        bool load_buffer(const std::string& path, buffer_t& buffer)
        {
                std::ifstream stream(path.c_str(), std::ios::binary | std::ios::in);
                if (!stream.is_open())
                {
                        return false;
                }

                return load_buffer_from_stream(stream, buffer);
        }

        bool save_string(std::ostream& ostream, const std::string& str)
        {
                std::cout << "save_string: size = " << str.size() << std::endl;

                ostream << str.size();

                for (std::size_t i = 0; i < str.size(); ++ i)
                {
                        ostream << str[i];
                }

                std::cout << "save_string: str = [" << str << "]" << std::endl;

                return ostream.good();
        }

        std::string load_string(std::istream& istream)
        {
                std::string::size_type size;
                istream >> size;

                std::cout << "load_string: size = " << size << std::endl;

                std::string str(size, ' ');
                for (std::size_t i = 0; i < str.size(); ++ i)
                {
                        istream >> str[i];
                }

                std::cout << "load_string: str = [" << str << "]" << std::endl;

                return str;
        }
}
