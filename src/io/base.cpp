#include "base.h"
#include <fstream>

namespace ncv
{
        bool io::load_skip(const data_t& data, size_t bytes, size_t& pos)
        {
                if (pos + bytes <= data.size())
                {
                        pos += bytes;
                        return true;
                }
                else
                {
                        return false;
                }
        }

        bool io::load_data(const data_t& data, size_t bytes, data_t& ldata, size_t& pos)
        {
                if (pos + bytes <= data.size())
                {
                        ldata.resize(bytes);
                        std::copy(data.data() + pos, data.data() + (pos + bytes), ldata.begin());
                        pos += bytes;
                        return true;
                }
                else
                {
                        return false;
                }
        }

        bool io::load_binary(std::istream& in, size_t bytes, data_t& data)
        {
                static const size_t chunk_size = 64 * 1024;
                char chunk[chunk_size];

                while (bytes > 0 && in)
                {
                        const std::streamsize to_read = bytes >= chunk_size ? chunk_size : bytes;
                        bytes -= to_read;

                        if (!in.read(chunk, to_read))
                        {
                                return false;
                        }

                        const std::streamsize count = in.gcount();
                        if (count == to_read)
                        {
                                data.insert(data.end(), chunk, chunk + count);
                        }
                        else
                        {
                                break;
                        }
                }

                return (bytes == std::string::npos) ? true : (bytes == data.size());
        }

        bool io::load_binary(std::istream& in, data_t& data)
        {
                return io::load_binary(in, std::string::npos, data);
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
