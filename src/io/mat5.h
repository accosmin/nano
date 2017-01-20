#pragma once

#include <ios>
#include "arch.h"
#include <vector>
#include <string>
#include <cstdint>
#include <functional>

namespace nano
{
        ///
        /// \brief data type
        ///
        enum class mat5_buffer_type : int
        {
                miINT8 = 1,
                miUINT8 = 2,
                miINT16 = 3,
                miUINT16 = 4,
                miINT32 = 5,
                miUINT32 = 6,
                miSINGLE = 7,
                miDOUBLE = 9,
                miINT64 = 12,
                miUINT64 = 13,
                miMATRIX = 14,
                miCOMPRESSED = 15,
                miUTF8 = 16,
                miUTF16 = 17,
                miUTF32 = 18,

                miUNKNOWN
        };

        ///
        /// \brief map a data type to string (logging purposes)
        ///
        NANO_PUBLIC std::string to_string(const mat5_buffer_type& type);

        ///
        /// \brief matlab5 header.
        ///
        struct mat5_header_t
        {
                ///
                /// \brief load from the input stream
                ///
                template <typename tstream>
                bool load(tstream& stream)
                {
                        return  stream.read(m_description, sizeof(m_description)) &&
                                stream.read(m_offset, sizeof(m_offset)) &&
                                stream.read(m_endian, sizeof(m_endian));
                }

                ///
                /// \brief header description as a string
                ///
                std::string description() const
                {
                        return std::string(m_description, m_description + sizeof(m_description));
                }

                // attributes
                char            m_description[116];
                char            m_offset[8];
                char            m_endian[4];
        };

        ///
        /// \brief matlab5 section.
        ///
        struct NANO_PUBLIC mat5_section_t
        {
                ///
                /// \brief constructor
                ///
                mat5_section_t();

                ///
                /// \brief load from the constants
                ///
                bool load(const uint32_t dtype, const uint32_t bytes);

                ///
                /// \brief load from the input stream
                ///
                template <typename tstream>
                bool load(tstream& istream)
                {
                        std::uint32_t dtype, bytes;
                        return  istream.read(reinterpret_cast<char*>(&dtype), sizeof(uint32_t)) &&
                                istream.read(reinterpret_cast<char*>(&bytes), sizeof(uint32_t)) &&
                                load(dtype, bytes);
                }

                // attributes
                std::streamsize         m_size;         ///< byte range of the whole section
                std::streamsize         m_dsize;        ///< byte range of the data section
                mat5_buffer_type        m_dtype;        ///< data type
        };

        NANO_PUBLIC std::ostream& operator<<(std::ostream&, const mat5_section_t&);

        /*
        ///
        /// \brief matlab5 multi-dimensional array consisting of multiple sections.
        ///
        struct NANO_PUBLIC mat5_array_t
        {
                ///
                /// \brief load from the input stream
                ///
                template <typename tstream>
                bool load(tstream& istream)
                {
                        mat5_section_t header;
                        if (    !header.load(istream) ||
                                header.m_dtype != mat5_buffer_type::miMATRIX)
                        {
                        }

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

                // attributes
                mat5_section_t          m_header;
                std::vector<size_t>     m_dims;                 ///< dimensions of the array
                std::string             m_name;                 ///< generic (Matlab) name
                mat5_section_t          m_meta_section;         ///<
                mat5_section_t          m_dims_section;         ///<
                mat5_section_t          m_name_section;         ///<
                mat5_section_t          m_data_section;         ///<
        };

        NANO_PUBLIC std::ostream& operator<<(std::ostream&, const mat5_array_t&);*/
}
