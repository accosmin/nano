#include "io_mat5.h"
#include "logger.h"
#include <fstream>
#include <limits>

namespace ncv
{
        /////////////////////////////////////////////////////////////////////////////////////////

        static u_int32_t make_uint32(const unsigned char* data)
        {
                return *reinterpret_cast<const u_int32_t*>(data);
        }

        /////////////////////////////////////////////////////////////////////////////////////////

        mat5::section_t::section_t(size_t begin)
                :       m_begin(begin), m_end(begin),
                        m_dbegin(begin), m_dend(begin),
                        m_dtype(data_type::miUNKNOWN)
        {
        }

        /////////////////////////////////////////////////////////////////////////////////////////

        bool mat5::section_t::load(size_t offset, size_t end, u_int32_t dtype, u_int32_t bytes)
        {
                // small data format
                if ((dtype >> 16) != 0)
                {
                        m_begin = offset;
                        m_end = offset + 8;

                        m_dbegin = offset + 4;
                        m_dend = offset + 8;

                        m_dtype = make_data_type((dtype << 16) >> 16);
                }

                // regular format
                else
                {
                        m_begin = offset;
                        m_end = offset + ((make_data_type(dtype) == data_type::miCOMPRESSED) ?
                                (8 + bytes) :
                                (8 + bytes + ((8 - bytes) % 8)));

                        m_dbegin = offset + 8;
                        m_dend = offset + 8 + bytes;

                        m_dtype = make_data_type(dtype);
                }

                return m_end <= end;
        }

        /////////////////////////////////////////////////////////////////////////////////////////

        bool mat5::section_t::load(std::ifstream& istream)
        {
                u_int32_t dtype, bytes;
                return  istream.read(reinterpret_cast<char*>(&dtype), sizeof(u_int32_t)) &&
                        istream.read(reinterpret_cast<char*>(&bytes), sizeof(u_int32_t)) &&
                        load(0, std::numeric_limits<size_t>::max(), dtype, bytes);
        }

        /////////////////////////////////////////////////////////////////////////////////////////

        bool mat5::section_t::load(const std::vector<u_int8_t>& data, size_t offset)
        {
                return  offset + 8 <= data.size() &&
                        load(offset, data.size(),
                             make_uint32(&data[offset + 0]), make_uint32(&data[offset + 4]));
        }

        /////////////////////////////////////////////////////////////////////////////////////////

        bool mat5::section_t::load(const std::vector<u_int8_t>& data, const section_t& prv)
        {
                return load(data, prv.m_end);
        }

        /////////////////////////////////////////////////////////////////////////////////////////

        mat5::array_t::array_t()
        {
        }

        /////////////////////////////////////////////////////////////////////////////////////////

        bool mat5::array_t::load(const std::vector<u_int8_t>& data)
        {
                // read & check header
                section_t header;
                if (!header.load(data))
                {
                        log_error() << "failed to load array!";
                        return false;
                }

                if (header.m_dtype != data_type::miMATRIX)
                {
                        log_error() << "invalid array type: expecting "
                                    << mat5::to_string(data_type::miMATRIX) << "!";
                        return false;
                }

                if (header.end() != data.size())
                {
                        log_error() << "invalid array size in bytes!";
                        return false;
                }

                log_info() << "array header: dtype = " << mat5::to_string(header.m_dtype)
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

                        log_info() << "array section: dtype = " << mat5::to_string(section.m_dtype)
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

        /////////////////////////////////////////////////////////////////////////////////////////

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

	/////////////////////////////////////////////////////////////////////////////////////////
}
