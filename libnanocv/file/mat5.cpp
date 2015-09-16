#include "mat5.h"
#include <limits>
#include <fstream>
#include <cstdint>
#include "libcore/logger.h"

namespace ncv
{
        using std::uint32_t;
        using mat5::buffer_type;

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
                buffer_type make_buffer_type(tint code)
                {
                        if (code == 1) return buffer_type::miINT8;
                        else if (code == 2) return buffer_type::miUINT8;
                        else if (code == 3) return buffer_type::miINT16;
                        else if (code == 4) return buffer_type::miUINT16;
                        else if (code == 5) return buffer_type::miINT32;
                        else if (code == 6) return buffer_type::miUINT32;
                        else if (code == 7) return buffer_type::miSINGLE;
                        else if (code == 9) return buffer_type::miDOUBLE;
                        else if (code == 12) return buffer_type::miINT64;
                        else if (code == 13) return buffer_type::miUINT64;
                        else if (code == 14) return buffer_type::miMATRIX;
                        else if (code == 15) return buffer_type::miCOMPRESSED;
                        else if (code == 16) return buffer_type::miUTF8;
                        else if (code == 17) return buffer_type::miUTF16;
                        else if (code == 18) return buffer_type::miUTF32;
                        else return buffer_type::miUNKNOWN;
                }
        }

        std::string mat5::to_string(const buffer_type& type)
        {
                if (type == buffer_type::miINT8) return "miINT8";
                else if (type == buffer_type::miUINT8) return "miUINT8";
                else if (type == buffer_type::miINT16) return "miINT16";
                else if (type == buffer_type::miUINT16) return "miUINT16";
                else if (type == buffer_type::miINT32) return "miINT32";
                else if (type == buffer_type::miUINT32) return "miUINT32";
                else if (type == buffer_type::miSINGLE) return "miSINGLE";
                else if (type == buffer_type::miDOUBLE) return "miDOUBLE";
                else if (type == buffer_type::miINT64) return "miINT64";
                else if (type == buffer_type::miUINT64) return "miUINT64";
                else if (type == buffer_type::miMATRIX) return "miMATRIX";
                else if (type == buffer_type::miCOMPRESSED) return "miCOMPRESSED";
                else if (type == buffer_type::miUTF8) return "miUTF8";
                else if (type == buffer_type::miUTF16) return "miUTF16";
                else if (type == buffer_type::miUTF32) return "miUTF32";
                else return "miUNKNOWN";
        }

        size_t mat5::to_bytes(const buffer_type& type)
        {
                if (type == buffer_type::miINT8) return 1;
                else if (type == buffer_type::miUINT8) return 1;
                else if (type == buffer_type::miINT16) return 2;
                else if (type == buffer_type::miUINT16) return 2;
                else if (type == buffer_type::miINT32) return 4;
                else if (type == buffer_type::miUINT32) return 4;
                else if (type == buffer_type::miSINGLE) return 4;
                else if (type == buffer_type::miDOUBLE) return 8;
                else if (type == buffer_type::miINT64) return 8;
                else if (type == buffer_type::miUINT64) return 8;
                else if (type == buffer_type::miMATRIX) return 0;
                else if (type == buffer_type::miCOMPRESSED) return 0;
                else if (type == buffer_type::miUTF8) return 0;
                else if (type == buffer_type::miUTF16) return 0;
                else if (type == buffer_type::miUTF32) return 0;
                else return 0;
        }

        mat5::section_t::section_t(size_t begin)
                :       m_begin(begin), m_end(begin),
                        m_dbegin(begin), m_dend(begin),
                        m_dtype(buffer_type::miUNKNOWN)
        {
        }

        bool mat5::section_t::load(size_t offset, size_t end, uint32_t dtype, uint32_t bytes)
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
                        m_end = offset + ((make_buffer_type(dtype) == buffer_type::miCOMPRESSED) ?
                                (8 + bytes) :
                                (8 + bytes + ((8 - bytes) % 8)));

                        m_dbegin = offset + 8;
                        m_dend = offset + 8 + bytes;

                        m_dtype = make_buffer_type(dtype);
                }

                return m_end <= end;
        }

        bool mat5::section_t::load(std::ifstream& istream)
        {
                uint32_t dtype, bytes;
                return  istream.read(reinterpret_cast<char*>(&dtype), sizeof(uint32_t)) &&
                        istream.read(reinterpret_cast<char*>(&bytes), sizeof(uint32_t)) &&
                        load(0, std::numeric_limits<size_t>::max(), dtype, bytes);
        }

        bool mat5::section_t::load(const io::buffer_t& data, size_t offset)
        {
                return  offset + 8 <= data.size() &&
                        load(offset, data.size(), make_uint32(&data[offset + 0]), make_uint32(&data[offset + 4]));
        }

        bool mat5::section_t::load(const io::buffer_t& data, const section_t& prv)
        {
                return  load(data, prv.m_end);
        }

        mat5::array_t::array_t()
        {
        }

        bool mat5::array_t::load(const io::buffer_t& data)
        {
                // read & check header
                section_t header;
                if (!header.load(data))
                {
                        log_error() << "failed to load array!";
                        return false;
                }

                if (header.m_dtype != buffer_type::miMATRIX)
                {
                        log_error() << "invalid array type: expecting "
                                    << to_string(buffer_type::miMATRIX) << "!";
                        return false;
                }

                if (header.end() != data.size())
                {
                        log_error() << "invalid array size in bytes!";
                        return false;
                }

                log_info() << "array header: dtype = " << to_string(header.m_dtype)
                           << ", bytes = " << header.size() << "/" << data.size() << ".";

                // read & check sections
                m_sections.clear();

                for (size_t i = 8; i < data.size(); )
                {
                        section_t section;
                        if (!section.load(data, i))
                        {
                                break;
                        }

                        log_info() << "array section: dtype = " << to_string(section.m_dtype)
                                   << ", range = [" << section.begin() << ", " << section.end()
                                   << "], drange = [" << section.dbegin() << ", " << section.dend()
                                   << "], bytes = " << section.dsize() << "/" << section.size() << ".";

                        m_sections.push_back(section);
                        i = section.end();
                }

                if (m_sections.size() != 4)
                {
                        log_error() << "invalid array sections! expecting 4 sections!";
                        return false;
                }

                // decode sections:
                //      first:  flags + class
                //      second: dimensions
                //      third:  name
//                                const mat::section_t& sect1 = m_sections[0];
                const section_t& sect2 = m_sections[1];
                const section_t& sect3 = m_sections[2];
                const section_t& sect4 = m_sections[3];

                m_name = std::string(data.begin() + sect3.dbegin(), data.begin() + sect3.dend());

                m_dims.clear();
                size_t values = 1;

                for (size_t i = sect2.dbegin(); i < sect2.dend(); i += 4)
                {
                        const size_t dim = make_uint32(&data[i]);
                        m_dims.push_back(dim);
                        values *= dim;
                }

                // check bytes
                if (values * to_bytes(sect4.m_dtype) != sect4.dsize())
                {
                        log_error() << "invalid array sections! mismatching number of bytes!";
                        return false;
                }

                // OK
                return true;
        }

        void mat5::array_t::log(logger_t& logger) const
        {
                logger << "sections = " << m_sections.size()
                       << ", name = " << m_name
                       << ", dims = ";
                for (size_t i = 0; i < m_dims.size(); i ++)
                {
                        logger << m_dims[i] << ((i + 1 == m_dims.size()) ? "" : "x");
                }
        }
}
