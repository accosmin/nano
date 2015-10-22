#pragma once

#include "mstream.h"
#include <vector>

namespace cortex
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
        /// \brief map a data type to string (logging issues)
        ///
        NANOCV_PUBLIC std::string to_string(const mat5_buffer_type& type);

        ///
        /// \brief section
        ///
        struct NANOCV_PUBLIC mat5_section_t
        {
                ///
                /// \brief constructor
                ///
                explicit mat5_section_t(std::streamsize begin = 0);

                bool load(std::streamsize offset, std::streamsize end, uint32_t dtype, uint32_t bytes);
                bool load(std::istream& stream);
                bool load(mstream_t& stream);

                // full section range
                std::streamsize begin() const { return m_begin; }
                std::streamsize end() const { return m_end; }
                std::streamsize size() const { return end() - begin(); }

                // section data range
                std::streamsize dbegin() const { return m_dbegin; }
                std::streamsize dend() const { return m_dend; }
                std::streamsize dsize() const { return dend() - dbegin(); }

                // attributes
                std::streamsize         m_begin, m_end;         ///< byte range of the whole section
                std::streamsize         m_dbegin, m_dend;       ///< byte range of the data section
                mat5_buffer_type        m_dtype;
        };

        template
        <
                typename tstream
        >
        tstream& operator<<(tstream& stream, const mat5_section_t& sect)
        {
                stream << "type = " << to_string(sect.m_dtype)
                       << ", range = [" << sect.begin() << ", " << sect.end() << "] = " << sect.size() << "B"
                       << ", data range = [" << sect.dbegin() << ", " << sect.dend() << "] = " << sect.dsize() << "B";
                return stream;
        }

        ///
        /// \brief read a multi-dimensional array consisting of multiple sections.
        ///
        struct NANOCV_PUBLIC mat5_array_t
        {
                ///
                /// \brief constructor
                ///
                mat5_array_t();

                ///
                /// \brief parse the array
                ///
                bool load(mstream_t& stream);

                // attributes
                std::vector<std::size_t>        m_dims;         ///< dimensions of the array
                std::string                     m_name;         ///< generic (Matlab) name
                std::vector<mat5_section_t>     m_sections;     ///< sections (dimensions, name, type, data)
        };

        template
        <
                typename tstream
        >
        tstream& operator<<(tstream& stream, const mat5_array_t& array)
        {
                stream << "sections = " << array.m_sections.size() << ", name = " << array.m_name << ", dims = ";
                for (std::size_t i = 0; i < array.m_dims.size(); i ++)
                {
                        stream << array.m_dims[i] << ((i + 1 == array.m_dims.size()) ? "" : "x");
                }
                return stream;
        }
}
