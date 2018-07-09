#pragma once

#include <cstdint>
#include "istream.h"
#include <functional>

namespace nano
{
        ///
        /// \brief data storage type.
        ///
        enum class mat5_dtype
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
        /// \brief data format type.
        ///
        enum class mat5_ftype
        {
                small,
                regular
        };

        ///
        /// \brief data chunk parent type.
        ///
        enum class mat5_ptype
        {
                none,
                miMATRIX
        };

        NANO_PUBLIC std::string to_string(const mat5_dtype);
        NANO_PUBLIC std::string to_string(const mat5_ftype);
        NANO_PUBLIC std::string to_string(const mat5_ptype);

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

        ///
        /// \brief matlab5 section.
        ///
        struct NANO_PUBLIC mat5_section_t
        {
                ///
                /// \brief constructor
                ///
                explicit mat5_section_t(const mat5_ptype ptype = mat5_ptype::none);

                ///
                /// \brief load from the input stream
                ///
                bool load(istream_t&);

                ///
                /// \brief skip past the data section
                ///
                bool skip(istream_t&) const;

                ///
                /// \brief check if the section is a corect sub-element of a matrix
                ///
                bool matrix_meta(istream_t&) const;
                bool matrix_name(istream_t&, std::string& name) const;
                bool matrix_dims(istream_t&, std::vector<int32_t>& dims) const;
                bool matrix_data(istream_t&) const;

                // attributes
                std::streamsize m_size{0};      ///< byte range of the whole section
                std::streamsize m_dsize{0};     ///< byte range of the data section
                mat5_dtype      m_dtype{mat5_dtype::miUNKNOWN}; ///<
                mat5_ftype      m_ftype{mat5_ftype::small};     ///<
                mat5_ptype      m_ptype{mat5_ptype::none};      ///< parent type (e.g. if a sub-section of a miMATRIX section)
                std::uint32_t   m_bytes{0};     ///< data section for small sections
        };

        NANO_PUBLIC std::ostream& operator<<(std::ostream&, const mat5_header_t&);
        NANO_PUBLIC std::ostream& operator<<(std::ostream&, const mat5_section_t&);

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
