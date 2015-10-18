#pragma once

#include "arch.h"
#include "buffer.h"

namespace cortex
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
                NANOCV_PUBLIC std::string to_string(const buffer_type& type);

                ///
                /// \brief map a data type to its size in bytes
                ///
                NANOCV_PUBLIC size_t to_bytes(const buffer_type& type);

                ///
                /// \brief section
                ///
                struct NANOCV_PUBLIC section_t
                {
                        ///
                        /// \brief constructor
                        ///
                        explicit section_t(size_t begin = 0);

                        bool load(size_t offset, size_t end, uint32_t dtype, uint32_t bytes);
                        bool load(std::ifstream& istream);
                        bool load(const buffer_t& data, size_t offset = 0);
                        bool load(const buffer_t& data, const section_t& prv);

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
                struct NANOCV_PUBLIC array_t
                {
                        ///
                        /// \brief constructor
                        ///
                        array_t();

                        ///
                        /// \brief parse the array
                        ///
                        bool load(const buffer_t& data);

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
