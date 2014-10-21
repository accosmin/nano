#pragma once

#include <string>
#include <vector>
#include <functional>
#include <iosfwd>
#include <cstring>

namespace ncv
{
        namespace io
        {
                using std::size_t;
                typedef std::vector<char>       data_t;

                ///
                /// \brief load POD structure from binary data
                ///
                template
                <
                        typename tstruct
                >
                bool load_struct(const data_t& data, tstruct& pod, size_t& pos)
                {
                        const size_t sizeof_pod = sizeof(pod);
                        if (pos + sizeof_pod < data.size())
                        {
                                memcpy((void*)&pod, (const void*)(data.data() + pos), sizeof_pod);
                                pos += sizeof_pod;
                                return true;
                        }
                        else
                        {
                                return false;
                        }
                }

                ///
                /// \brief load binary file in memory
                ///
                bool load_binary(std::istream& in, size_t bytes, data_t& data);
                bool load_binary(std::istream& in, data_t& data);
                bool load_binary(const std::string& path, data_t& data);

                ///
                /// \brief save memory buffer to binary file
                ///
                bool save_binary(const data_t& data, const std::string& path);

                ///
                /// \brief callback to execute when a file was decompressed from an archive
                ///     - (filename, uncompressed file content loaded in memory)
                ///
                typedef std::function<void(const std::string&, const data_t&)> data_callback_t;
        }
}
