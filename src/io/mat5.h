#pragma once

#include <cstdint>
#include "istream.h"
#include <functional>

namespace nano
{
        ///
        /// \brief data type
        ///
        enum class mat5_data_type
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

        enum class mat5_format_type
        {
                small,
                regular
        };

        NANO_PUBLIC std::string to_string(const mat5_data_type);
        NANO_PUBLIC std::string to_string(const mat5_format_type);

        ///
        /// \brief matlab5 header.
        ///
        struct NANO_PUBLIC mat5_header_t
        {
                ///
                /// \brief load from the input stream
                ///
                bool load(istream_t&);

                ///
                /// \brief header description as a string
                ///
                std::string description() const;

                // attributes
                char            m_description[116];
                char            m_offset[8];
                char            m_endian[4];
        };

        NANO_PUBLIC std::ostream& operator<<(std::ostream&, const mat5_header_t&);

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
                bool load(istream_t&);

                // attributes
                std::streamsize         m_size;         ///< byte range of the whole section
                std::streamsize         m_dsize;        ///< byte range of the data section
                mat5_data_type          m_dtype;        ///<
                mat5_format_type        m_ftype;        ///<
        };

        NANO_PUBLIC std::ostream& operator<<(std::ostream&, const mat5_section_t&);

        ///
        /// \brief matlab5 multi-dimensional array consisting of multiple sections.
        ///
        struct NANO_PUBLIC mat5_array_t
        {
                ///
                /// \brief load from the input stream
                ///
                bool load(istream_t& istream);

                // attributes
                mat5_section_t          m_header;
                std::vector<size_t>     m_dims;                 ///< dimensions of the array
                std::string             m_name;                 ///< generic (Matlab) name
                mat5_section_t          m_meta_section;         ///<
                mat5_section_t          m_dims_section;         ///<
                mat5_section_t          m_name_section;         ///<
                mat5_section_t          m_data_section;         ///<
        };

        ///
        /// \brief callback to execute when the header is decoded
        ///     - returns true if it should continue
        ///
        using mat5_header_callback_t = std::function<bool(const mat5_header_t&)>;

        ///
        /// \brief callback to execute when a section is decoded
        ///     (section description, stream to read the data)
        ///     - returns true if it should continue
        ///
        using mat5_section_callback_t = std::function<bool(const mat5_section_t&, istream_t&)>;

        ///
        /// \brief callback to execute when an error was detected
        ///     - (error message)
        ///
        using mat5_error_callback_t = std::function<void(const std::string&)>;

        ///
        /// \brief decode a matlab5 file (.mat)
        ///
        NANO_PUBLIC bool load_mat5(const std::string& path,
                const mat5_header_callback_t&, const mat5_section_callback_t&, const mat5_error_callback_t&);
}
