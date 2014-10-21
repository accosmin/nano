#include "io.h"
#include <fstream>

namespace ncv
{
        bool io::load_binary(std::istream& in, size_t bytes, data_t& data)
        {
                static const size_t chunk_size = 16 * 1024;
                char chunk[chunk_size];

                while (bytes > 0)
                {
                        const std::streamsize to_read = bytes >= chunk_size ? chunk_size : bytes;
                        bytes -= to_read;

                        const std::streamsize count = in.readsome(chunk, to_read);
                        if (count > 0)
                        {
                                data.insert(data.end(), chunk, chunk + count);
                        }
                        else
                        {
                                return false;
                        }
                }

                return true;
        }

        bool io::load_binary(std::istream& in, data_t& data)
        {
                static const std::streamsize chunk_size = 16 * 1024;
                char chunk[chunk_size];

                std::streamsize count;
                while ((count = in.readsome(chunk, chunk_size)) > 0)
                {
                        data.insert(data.end(), chunk, chunk + count);
                }

                return true;
        }

        bool io::load_binary(const std::string& path, data_t& data)
        {
                std::ifstream in(path.c_str(), std::ios_base::binary | std::ios_base::in);

                return  in.is_open() &&
                        load_binary(in, data);
        }

        bool io::save_binary(const data_t& data, const std::string& path)
        {
                std::ofstream out(path.c_str(), std::ios_base::binary | std::ios_base::out);

                return  out.is_open() &&
                        out.write(data.data(), data.size());
        }
}
