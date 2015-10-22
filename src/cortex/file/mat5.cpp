#include "mat5.h"
#include <fstream>

namespace cortex
{
        using std::uint32_t;

        namespace
        {
                uint32_t make_uint32(const char* data)
                {
                        return *reinterpret_cast<const uint32_t*>(data);
                }

                template
                <
                        typename tint
                >
                mat5_buffer_type make_buffer_type(tint code)
                {
                        if (code == 1) return mat5_buffer_type::miINT8;
                        else if (code == 2) return mat5_buffer_type::miUINT8;
                        else if (code == 3) return mat5_buffer_type::miINT16;
                        else if (code == 4) return mat5_buffer_type::miUINT16;
                        else if (code == 5) return mat5_buffer_type::miINT32;
                        else if (code == 6) return mat5_buffer_type::miUINT32;
                        else if (code == 7) return mat5_buffer_type::miSINGLE;
                        else if (code == 9) return mat5_buffer_type::miDOUBLE;
                        else if (code == 12) return mat5_buffer_type::miINT64;
                        else if (code == 13) return mat5_buffer_type::miUINT64;
                        else if (code == 14) return mat5_buffer_type::miMATRIX;
                        else if (code == 15) return mat5_buffer_type::miCOMPRESSED;
                        else if (code == 16) return mat5_buffer_type::miUTF8;
                        else if (code == 17) return mat5_buffer_type::miUTF16;
                        else if (code == 18) return mat5_buffer_type::miUTF32;
                        else return mat5_buffer_type::miUNKNOWN;
                }

                std::streamsize to_bytes(const mat5_buffer_type& type)
                {
                        if (type == mat5_buffer_type::miINT8) return 1;
                        else if (type == mat5_buffer_type::miUINT8) return 1;
                        else if (type == mat5_buffer_type::miINT16) return 2;
                        else if (type == mat5_buffer_type::miUINT16) return 2;
                        else if (type == mat5_buffer_type::miINT32) return 4;
                        else if (type == mat5_buffer_type::miUINT32) return 4;
                        else if (type == mat5_buffer_type::miSINGLE) return 4;
                        else if (type == mat5_buffer_type::miDOUBLE) return 8;
                        else if (type == mat5_buffer_type::miINT64) return 8;
                        else if (type == mat5_buffer_type::miUINT64) return 8;
                        else if (type == mat5_buffer_type::miMATRIX) return 0;
                        else if (type == mat5_buffer_type::miCOMPRESSED) return 0;
                        else if (type == mat5_buffer_type::miUTF8) return 0;
                        else if (type == mat5_buffer_type::miUTF16) return 0;
                        else if (type == mat5_buffer_type::miUTF32) return 0;
                        else return 0;
                }
        }

        std::string to_string(const mat5_buffer_type& type)
        {
                if (type == mat5_buffer_type::miINT8) return "miINT8";
                else if (type == mat5_buffer_type::miUINT8) return "miUINT8";
                else if (type == mat5_buffer_type::miINT16) return "miINT16";
                else if (type == mat5_buffer_type::miUINT16) return "miUINT16";
                else if (type == mat5_buffer_type::miINT32) return "miINT32";
                else if (type == mat5_buffer_type::miUINT32) return "miUINT32";
                else if (type == mat5_buffer_type::miSINGLE) return "miSINGLE";
                else if (type == mat5_buffer_type::miDOUBLE) return "miDOUBLE";
                else if (type == mat5_buffer_type::miINT64) return "miINT64";
                else if (type == mat5_buffer_type::miUINT64) return "miUINT64";
                else if (type == mat5_buffer_type::miMATRIX) return "miMATRIX";
                else if (type == mat5_buffer_type::miCOMPRESSED) return "miCOMPRESSED";
                else if (type == mat5_buffer_type::miUTF8) return "miUTF8";
                else if (type == mat5_buffer_type::miUTF16) return "miUTF16";
                else if (type == mat5_buffer_type::miUTF32) return "miUTF32";
                else return "miUNKNOWN";
        }

        mat5_section_t::mat5_section_t(std::streamsize begin)
                :       m_begin(begin), m_end(begin),
                        m_dbegin(begin), m_dend(begin),
                        m_dtype(mat5_buffer_type::miUNKNOWN)
        {
        }

        bool mat5_section_t::load(std::streamsize offset, uint32_t dtype, uint32_t bytes)
        {
                // small data format
                if ((dtype >> 16) != 0)
                {
                        m_begin = offset;
                        m_end = offset + 8;

                        m_dbegin = offset + 4;
                        m_dend = offset + 8;

                        m_dtype = make_buffer_type((dtype << 16) >> 16);
                }

                // regular format
                else
                {
                        m_begin = offset;
                        m_end = offset + ((make_buffer_type(dtype) == mat5_buffer_type::miCOMPRESSED) ?
                                (8 + bytes) :
                                (8 + bytes + ((8 - bytes) % 8)));

                        m_dbegin = offset + 8;
                        m_dend = offset + 8 + bytes;

                        m_dtype = make_buffer_type(dtype);
                }

                return true;
        }

        bool mat5_array_t::load_header(mstream_t& istream)
        {
                mat5_section_t header;
                return  header.load(istream) &&
                        header.m_dtype == mat5_buffer_type::miMATRIX &&
                        header.end() == istream.size();
        }

        bool mat5_array_t::load_body(mstream_t& istream)
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
}
