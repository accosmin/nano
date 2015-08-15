#pragma once

#include "buffer.h"

namespace ncv
{
        class logger_t;

        namespace mat5
        {
                using std::size_t;

                ///
                /// \brief data type
                ///
                enum class buffer_type : int
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
                inline std::string to_string(const buffer_type& type)
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

                ///
                /// \brief map a data type to its size in bytes
                ///
                inline size_t to_bytes(const buffer_type& type)
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

                ///
                /// \brief map an integer to a data type
                ///
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

                ///
                /// \brief section
                ///
                struct section_t
                {
                        ///
                        /// \brief constructor
                        ///
                        explicit section_t(size_t begin = 0);

                        bool load(size_t offset, size_t end, uint32_t dtype, uint32_t bytes);
                        bool load(std::ifstream& istream);
                        bool load(const io::buffer_t& data, size_t offset = 0);
                        bool load(const io::buffer_t& data, const section_t& prv);

                        // full section range
                        size_t begin() const { return m_begin; }
                        size_t end() const { return m_end; }
                        size_t size() const { return end() - begin(); }

                        // section data range
                        size_t dbegin() const { return m_dbegin; }
                        size_t dend() const { return m_dend; }
                        size_t dsize() const { return dend() - dbegin(); }

                        // attributes
                        size_t                  m_begin, m_end;         ///< byte range of the whole section
                        size_t                  m_dbegin, m_dend;       ///< byte range of the data section
                        buffer_type               m_dtype;
                };

                typedef std::vector<section_t>  sections_t;

                ///
                /// \brief read a multi-dimensional array consisting of multiple sections.
                ///
                struct array_t
                {
                        ///
                        /// \brief constructor
                        ///
                        array_t();

                        ///
                        /// \brief parse the array
                        ///
                        bool load(const io::buffer_t& data);

                        ///
                        /// \brief describe the array
                        ///
                        void log(logger_t& logger) const;

                        // attributes
                        std::vector<size_t>     m_dims;                 ///< dimensions of the array
                        std::string             m_name;                 ///< generic (Matlab) name
                        sections_t              m_sections;             ///< sections (dimensions, name, type, data)
                };
        }
}
