#include "mat5.h"
#include "imstream.h"
#include <fstream>

namespace nano
{
        using std::uint32_t;

        namespace
        {
                uint32_t make_uint32(const char* data)
                {
                        return *reinterpret_cast<const uint32_t*>(data);
                }

                template <typename tinteger>
                mat5_buffer_type make_buffer_type(const tinteger code)
                {
                        switch (code)
                        {
                        case 1:         return mat5_buffer_type::miINT8;
                        case 2:         return mat5_buffer_type::miUINT8;
                        case 3:         return mat5_buffer_type::miINT16;
                        case 4:         return mat5_buffer_type::miUINT16;
                        case 5:         return mat5_buffer_type::miINT32;
                        case 6:         return mat5_buffer_type::miUINT32;
                        case 7:         return mat5_buffer_type::miSINGLE;
                        case 9:         return mat5_buffer_type::miDOUBLE;
                        case 12:        return mat5_buffer_type::miINT64;
                        case 13:        return mat5_buffer_type::miUINT64;
                        case 14:        return mat5_buffer_type::miMATRIX;
                        case 15:        return mat5_buffer_type::miCOMPRESSED;
                        case 16:        return mat5_buffer_type::miUTF8;
                        case 17:        return mat5_buffer_type::miUTF16;
                        case 18:        return mat5_buffer_type::miUTF32;
                        default:        return mat5_buffer_type::miUNKNOWN;
                        }
                }

                std::streamsize to_bytes(const mat5_buffer_type& type)
                {
                        switch (type)
                        {
                        case mat5_buffer_type::miINT8:          return 1;
                        case mat5_buffer_type::miUINT8:         return 1;
                        case mat5_buffer_type::miINT16:         return 2;
                        case mat5_buffer_type::miUINT16:        return 2;
                        case mat5_buffer_type::miINT32:         return 4;
                        case mat5_buffer_type::miUINT32:        return 4;
                        case mat5_buffer_type::miSINGLE:        return 4;
                        case mat5_buffer_type::miDOUBLE:        return 8;
                        case mat5_buffer_type::miINT64:         return 8;
                        case mat5_buffer_type::miUINT64:        return 8;
                        case mat5_buffer_type::miMATRIX:        return 0;
                        case mat5_buffer_type::miCOMPRESSED:    return 0;
                        case mat5_buffer_type::miUTF8:          return 0;
                        case mat5_buffer_type::miUTF16:         return 0;
                        case mat5_buffer_type::miUTF32:         return 0;
                        default:                                return 0;
                        }
                }
        }

        std::string to_string(const mat5_buffer_type& type)
        {
                switch (type)
                {
                case mat5_buffer_type::miINT8:                  return "miINT8";
                case mat5_buffer_type::miUINT8:                 return "miUINT8";
                case mat5_buffer_type::miINT16:                 return "miINT16";
                case mat5_buffer_type::miUINT16:                return "miUINT16";
                case mat5_buffer_type::miINT32:                 return "miINT32";
                case mat5_buffer_type::miUINT32:                return "miUINT32";
                case mat5_buffer_type::miSINGLE:                return "miSINGLE";
                case mat5_buffer_type::miDOUBLE:                return "miDOUBLE";
                case mat5_buffer_type::miINT64:                 return "miINT64";
                case mat5_buffer_type::miUINT64:                return "miUINT64";
                case mat5_buffer_type::miMATRIX:                return "miMATRIX";
                case mat5_buffer_type::miCOMPRESSED:            return "miCOMPRESSED";
                case mat5_buffer_type::miUTF8:                  return "miUTF8";
                case mat5_buffer_type::miUTF16:                 return "miUTF16";
                case mat5_buffer_type::miUTF32:                 return "miUTF32";
                default:                                        return "miUNKNOWN";
                }
        }

        mat5_section_t::mat5_section_t() :
                m_size(0),
                m_dsize(0),
                m_dtype(mat5_buffer_type::miUNKNOWN)
        {
        }

        bool mat5_section_t::load(const uint32_t dtype, const uint32_t bytes)
        {
                // small data format
                if ((dtype >> 16) != 0)
                {
                        m_size = 8;
                        m_dsize = 4;
                        m_dtype = make_buffer_type((dtype << 16) >> 16);
                }

                // regular format
                else
                {
                        const auto compressed = make_buffer_type(dtype) == mat5_buffer_type::miCOMPRESSED;
                        m_size = compressed ?
                                (8 + bytes) :
                                (8 + bytes + static_cast<uint32_t>((7 * static_cast<uint64_t>(bytes)) % 8));
                        m_dsize = bytes;
                        m_dtype = make_buffer_type(dtype);
                }

                return true;
        }

        std::ostream& operator<<(std::ostream& ostream, const mat5_section_t& sect)
        {
                ostream << "type = " << to_string(sect.m_dtype)
                        << ", size = " << sect.m_size << "B"
                        << ", data size = " << sect.m_dsize << "B";
                return ostream;
        }

        bool mat5_array_t::load_header(imstream_t& istream)
        {
                mat5_section_t header;
                return  header.load(istream) &&
                        header.m_dtype == mat5_buffer_type::miMATRIX;
        }

        bool mat5_array_t::load_body(imstream_t& istream)
        {
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
        }

        std::ostream& operator<<(std::ostream& ostream, const mat5_array_t& array)
        {
                ostream << "sections = " << array.m_sections.size() << ", name = " << array.m_name << ", dims = ";
                for (std::size_t i = 0; i < array.m_dims.size(); ++ i)
                {
                        ostream << array.m_dims[i] << ((i + 1 == array.m_dims.size()) ? "" : "x");
                }
                return ostream;
        }
}
