#pragma once

#include <ios>
#include "arch.h"
#include <vector>
#include <string>
#include <cstdint>
#include <functional>

namespace nano
{
        class imstream_t;

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
                template <typename tstream>
                bool load(tstream& stream)
                {
                        return  stream.read(m_description, sizeof(m_description)) &&
                                stream.read(m_offset, sizeof(m_offset)) &&
                                stream.read(m_endian, sizeof(m_endian));
                }

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
                explicit mat5_section_t(std::streamsize begin = 0);

                ///
                /// \brief load from the constants
                ///
                bool load(std::streamsize offset, uint32_t dtype, uint32_t bytes);

                ///
                /// \brief load from the input stream
                ///
                template <typename tstream>
                bool load(tstream& istream)
                {
                        const auto offset = istream.tellg();

                        std::uint32_t dtype, bytes;
                        return  istream.read(reinterpret_cast<char*>(&dtype), sizeof(uint32_t)) &&
                                istream.read(reinterpret_cast<char*>(&bytes), sizeof(uint32_t)) &&
                                load(offset, dtype, bytes);
                }

                /// full section range
                std::streamsize begin() const { return m_begin; }
                std::streamsize end() const { return m_end; }
                std::streamsize size() const { return end() - begin(); }

                /// section data range
                std::streamsize dbegin() const { return m_dbegin; }
                std::streamsize dend() const { return m_dend; }
                std::streamsize dsize() const { return dend() - dbegin(); }

                // attributes
                std::streamsize         m_begin, m_end;         ///< byte range of the whole section
                std::streamsize         m_dbegin, m_dend;       ///< byte range of the data section
                mat5_buffer_type        m_dtype;
        };

        NANO_PUBLIC std::ostream& operator<<(std::ostream&, const mat5_section_t&);

        ///
        /// \brief multi-dimensional array consisting of multiple sections
        ///
        struct NANO_PUBLIC mat5_array_t
        {
                ///
                /// \brief load header section from the input stream
                ///
                static bool load_header(imstream_t& istream);

                ///
                /// \brief load body sections from the input stream
                ///
                bool load_body(imstream_t& istream);

                // attributes
                std::vector<std::size_t>        m_dims;         ///< dimensions of the array
                std::string                     m_name;         ///< generic (Matlab) name
                std::vector<mat5_section_t>     m_sections;     ///< sections (dimensions, name, type, data)
        };

        NANO_PUBLIC std::ostream& operator<<(std::ostream&, const mat5_array_t&);
}
