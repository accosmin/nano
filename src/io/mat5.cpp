#include "mat5.h"
#include <ostream>
#include <fstream>
#include "istream_std.h"
#include "istream_zlib.h"

namespace nano
{
        using std::uint32_t;

        inline uint32_t make_uint32(const char* data)
        {
                return *reinterpret_cast<const uint32_t*>(data);
        }

        template <typename tinteger>
        inline mat5_data_type make_data_type(const tinteger code)
        {
                switch (code)
                {
                case 1:         return mat5_data_type::miINT8;
                case 2:         return mat5_data_type::miUINT8;
                case 3:         return mat5_data_type::miINT16;
                case 4:         return mat5_data_type::miUINT16;
                case 5:         return mat5_data_type::miINT32;
                case 6:         return mat5_data_type::miUINT32;
                case 7:         return mat5_data_type::miSINGLE;
                case 9:         return mat5_data_type::miDOUBLE;
                case 12:        return mat5_data_type::miINT64;
                case 13:        return mat5_data_type::miUINT64;
                case 14:        return mat5_data_type::miMATRIX;
                case 15:        return mat5_data_type::miCOMPRESSED;
                case 16:        return mat5_data_type::miUTF8;
                case 17:        return mat5_data_type::miUTF16;
                case 18:        return mat5_data_type::miUTF32;
                default:        return mat5_data_type::miUNKNOWN;
                }
        }

        inline std::streamsize to_bytes(const mat5_data_type& type)
        {
                switch (type)
                {
                case mat5_data_type::miINT8:            return 1;
                case mat5_data_type::miUINT8:           return 1;
                case mat5_data_type::miINT16:           return 2;
                case mat5_data_type::miUINT16:          return 2;
                case mat5_data_type::miINT32:           return 4;
                case mat5_data_type::miUINT32:          return 4;
                case mat5_data_type::miSINGLE:          return 4;
                case mat5_data_type::miDOUBLE:          return 8;
                case mat5_data_type::miINT64:           return 8;
                case mat5_data_type::miUINT64:          return 8;
                case mat5_data_type::miMATRIX:          return 0;
                case mat5_data_type::miCOMPRESSED:      return 0;
                case mat5_data_type::miUTF8:            return 0;
                case mat5_data_type::miUTF16:           return 0;
                case mat5_data_type::miUTF32:           return 0;
                default:                                return 0;
                }
        }

        std::string to_string(const mat5_data_type type)
        {
                switch (type)
                {
                case mat5_data_type::miINT8:            return "miINT8";
                case mat5_data_type::miUINT8:           return "miUINT8";
                case mat5_data_type::miINT16:           return "miINT16";
                case mat5_data_type::miUINT16:          return "miUINT16";
                case mat5_data_type::miINT32:           return "miINT32";
                case mat5_data_type::miUINT32:          return "miUINT32";
                case mat5_data_type::miSINGLE:          return "miSINGLE";
                case mat5_data_type::miDOUBLE:          return "miDOUBLE";
                case mat5_data_type::miINT64:           return "miINT64";
                case mat5_data_type::miUINT64:          return "miUINT64";
                case mat5_data_type::miMATRIX:          return "miMATRIX";
                case mat5_data_type::miCOMPRESSED:      return "miCOMPRESSED";
                case mat5_data_type::miUTF8:            return "miUTF8";
                case mat5_data_type::miUTF16:           return "miUTF16";
                case mat5_data_type::miUTF32:           return "miUTF32";
                default:                                return "miUNKNOWN";
                }
        }

        std::string to_string(const mat5_format_type type)
        {
                switch (type)
                {
                case mat5_format_type::small:           return "small";
                case mat5_format_type::regular:         return "regular";
                default:                                return "unknown";
                }
        }

        bool mat5_header_t::load(istream_t& stream)
        {
                return  stream.read(m_description) &&
                        stream.read(m_offset) &&
                        stream.read(m_endian);
        }

        std::string mat5_header_t::description() const
        {
                return std::string(m_description, m_description + sizeof(m_description));
        }

        mat5_section_t::mat5_section_t() :
                m_size(0),
                m_dsize(0),
                m_dtype(mat5_data_type::miUNKNOWN),
                m_ftype(mat5_format_type::small)
        {
        }

        bool mat5_section_t::load(const uint32_t dtype, const uint32_t bytes)
        {
                // small data format
                if ((dtype >> 16) != 0)
                {
                        m_size = 8;
                        m_dsize = 4;
                        m_dtype = make_data_type((dtype << 16) >> 16);
                        m_ftype = mat5_format_type::small;
                }

                // regular format
                else
                {
                        const auto compressed = make_data_type(dtype) == mat5_data_type::miCOMPRESSED;
                        m_size = compressed ?
                                (8 + bytes) :
                                (8 + bytes + static_cast<uint32_t>((7 * static_cast<uint64_t>(bytes)) % 8));
                        m_dsize = bytes;
                        m_dtype = make_data_type(dtype);
                        m_ftype = mat5_format_type::regular;
                }

                return true;
        }

        bool mat5_section_t::load(istream_t& stream)
        {
                std::uint32_t dtype, bytes;
                return  stream.read(dtype) &&
                        stream.read(bytes) &&
                        load(dtype, bytes);
        }

        std::ostream& operator<<(std::ostream& ostream, const mat5_section_t& sect)
        {
                ostream << "type = " << to_string(sect.m_dtype)
                        << ", format = " << to_string(sect.m_ftype)
                        << ", size = " << sect.m_size << "B"
                        << ", data size = " << sect.m_dsize << "B";
                return ostream;
        }

        bool mat5_array_t::load(istream_t& stream)
        {
                if (    !m_header.load(stream) ||
                        m_header.m_dtype != mat5_data_type::miMATRIX)
                {
                        return false;
                }

                //
                if (!m_meta_section.load(stream))
                {
                        return false;
                }

                return true;

                /*

                // read & check sections
                m_sections.clear();

                mat5_section_t section;
                while (istream && m_sections.size() < 5 && section.load(istream))
                {
                        m_sections.push_back(section);
                        istream.seekg(section.end());   // move past the data section to read the next section
                }

                if (m_sections.size() != 4)
                {
                        return false;
                }

                // decode sections:
                //      first:  flags + class
                //      second: dimensions
                //      third:  name
                //      fourth: data matrix/tensor
//                const mat5_section_t& sect1 = m_sections[0];
                const mat5_section_t& sect2 = m_sections[1];
                const mat5_section_t& sect3 = m_sections[2];
                const mat5_section_t& sect4 = m_sections[3];

                m_name = std::string(istream.data() + sect3.dbegin(),
                                     istream.data() + sect3.dend());

                m_dims.clear();
                std::streamsize values = 1;
                for (std::streamsize i = sect2.dbegin(); i < sect2.dend(); i += 4)
                {
                        const auto dim = make_uint32(&istream.data()[i]);
                        m_dims.push_back(dim);
                        values *= dim;
                }

                // check bytes
                return values * to_bytes(sect4.m_dtype) == sect4.dsize();
                */
        }

        std::ostream& operator<<(std::ostream& ostream, const mat5_array_t& array)
        {
                ostream << "name = " << array.m_name << ", dims = ";
                for (std::size_t i = 0; i < array.m_dims.size(); ++ i)
                {
                        ostream << array.m_dims[i] << ((i + 1 == array.m_dims.size()) ? "" : "x");
                }
                return ostream;
        }

        static bool load_mat5(istream_t& stream,
                const mat5_section_callback_t& scallback,
                const mat5_error_callback_t& ecallback)
        {
                while (stream)
                {
                        mat5_section_t section;
                        if (!section.load(stream))
                        {
                                ecallback("failed to load section!");
                                return false;
                        }

                        switch (section.m_dtype)
                        {
                        case mat5_data_type::miCOMPRESSED:
                                {
                                        // gzip compressed section
                                        zlib_istream_t zstream(stream, section.m_dsize);
                                        if (!load_mat5(zstream, scallback, ecallback))
                                        {
                                                return false;
                                        }
                                }
                                break;

                        case mat5_data_type::miMATRIX:
                                {
                                        // array/matrix section, so read the sub-elements
                                        if (!load_mat5(stream, scallback, ecallback))
                                        {
                                                return false;
                                        }
                                }
                                break;

                        default:
                                if (!scallback(section, stream))
                                {
                                        return false;
                                }
                        }
                }

                return true;
        }

        bool load_mat5(const std::string& path,
                const mat5_header_callback_t& hcallback,
                const mat5_section_callback_t& scallback,
                const mat5_error_callback_t& ecallback)
        {
                std::ifstream istream(path.c_str(), std::ios::binary | std::ios::in);
                if (!istream.is_open())
                {
                        ecallback("failed to open file <" + path + ">!");
                        return false;
                }

                std_istream_t stream(istream);

                // load header
                mat5_header_t header;
                if (!header.load(stream))
                {
                        ecallback("failed to load header!");
                        return false;
                }
                if (!hcallback(header))
                {
                        return false;
                }

                // load sections
                return load_mat5(stream, scallback, ecallback);
        }
}
