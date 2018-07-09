#include "mat5.h"
#include <ostream>
#include <fstream>
#include "istream_mem.h"
#include "istream_std.h"
#include "istream_zlib.h"

using namespace nano;

using std::uint32_t;

inline uint32_t make_uint32(const char* data)
{
        return *reinterpret_cast<const uint32_t*>(data);
}

template <typename tinteger>
inline mat5_dtype make_dtype(const tinteger code)
{
        switch (code)
        {
        case 1:         return mat5_dtype::miINT8;
        case 2:         return mat5_dtype::miUINT8;
        case 3:         return mat5_dtype::miINT16;
        case 4:         return mat5_dtype::miUINT16;
        case 5:         return mat5_dtype::miINT32;
        case 6:         return mat5_dtype::miUINT32;
        case 7:         return mat5_dtype::miSINGLE;
        case 9:         return mat5_dtype::miDOUBLE;
        case 12:        return mat5_dtype::miINT64;
        case 13:        return mat5_dtype::miUINT64;
        case 14:        return mat5_dtype::miMATRIX;
        case 15:        return mat5_dtype::miCOMPRESSED;
        case 16:        return mat5_dtype::miUTF8;
        case 17:        return mat5_dtype::miUTF16;
        case 18:        return mat5_dtype::miUTF32;
        default:        return mat5_dtype::miUNKNOWN;
        }
}

inline std::streamsize to_bytes(const mat5_dtype& type)
{
        switch (type)
        {
        case mat5_dtype::miINT8:            return 1;
        case mat5_dtype::miUINT8:           return 1;
        case mat5_dtype::miINT16:           return 2;
        case mat5_dtype::miUINT16:          return 2;
        case mat5_dtype::miINT32:           return 4;
        case mat5_dtype::miUINT32:          return 4;
        case mat5_dtype::miSINGLE:          return 4;
        case mat5_dtype::miDOUBLE:          return 8;
        case mat5_dtype::miINT64:           return 8;
        case mat5_dtype::miUINT64:          return 8;
        case mat5_dtype::miMATRIX:          return 0;
        case mat5_dtype::miCOMPRESSED:      return 0;
        case mat5_dtype::miUTF8:            return 0;
        case mat5_dtype::miUTF16:           return 0;
        case mat5_dtype::miUTF32:           return 0;
        default:                                return 0;
        }
}

std::string nano::to_string(const mat5_dtype type)
{
        switch (type)
        {
        case mat5_dtype::miINT8:            return "miINT8";
        case mat5_dtype::miUINT8:           return "miUINT8";
        case mat5_dtype::miINT16:           return "miINT16";
        case mat5_dtype::miUINT16:          return "miUINT16";
        case mat5_dtype::miINT32:           return "miINT32";
        case mat5_dtype::miUINT32:          return "miUINT32";
        case mat5_dtype::miSINGLE:          return "miSINGLE";
        case mat5_dtype::miDOUBLE:          return "miDOUBLE";
        case mat5_dtype::miINT64:           return "miINT64";
        case mat5_dtype::miUINT64:          return "miUINT64";
        case mat5_dtype::miMATRIX:          return "miMATRIX";
        case mat5_dtype::miCOMPRESSED:      return "miCOMPRESSED";
        case mat5_dtype::miUTF8:            return "miUTF8";
        case mat5_dtype::miUTF16:           return "miUTF16";
        case mat5_dtype::miUTF32:           return "miUTF32";
        default:                                return "miUNKNOWN";
        }
}

std::string nano::to_string(const mat5_ftype type)
{
        switch (type)
        {
        case mat5_ftype::small:           return "small";
        case mat5_ftype::regular:         return "regular";
        default:                                return "unknown";
        }
}

std::string nano::to_string(const mat5_ptype type)
{
        switch (type)
        {
        case mat5_ptype::none:            return ".";
        case mat5_ptype::miMATRIX:        return "miMATRIX";
        default:                                return "unknown";
        }
}

bool mat5_header_t::load(istream_t& stream)
{
        return  stream.read(m_description) &&
                stream.read(m_offset) &&
                stream.read(m_endian);
}

std::string mat5_header_t::description() const
{
        return std::string(m_description, m_description + sizeof(m_description));
}

template <typename tdata, typename tstream, typename tvalue>
static bool read_vector(tstream&& stream,
        std::streamsize bytes,
        std::vector<tvalue>& values)
{
        const auto value_size = static_cast<std::streamsize>(sizeof(tvalue));
        values.resize(static_cast<size_t>(bytes / value_size));
        return  sizeof(tdata) == sizeof(tvalue) &&
                (bytes % value_size == 0) &&
                stream.read(reinterpret_cast<char*>(values.data()), bytes) == bytes;
}

template <typename tstream, typename tvalue>
static bool read_vector(tstream&& stream,
        const mat5_dtype dtype, const std::streamsize dsize,
        std::vector<tvalue>& values)
{
        switch (dtype)
        {
        case mat5_dtype::miINT8:    return read_vector<int8_t>(stream, dsize, values);
        case mat5_dtype::miINT32:   return read_vector<int32_t>(stream, dsize, values);
        default:                        return false;
        }
}

template <typename tstream, typename tvalue>
static bool read_vector(tstream&& stream,
        const mat5_ftype ftype, const uint32_t bytes, const mat5_dtype dtype, const std::streamsize dsize,
        std::vector<tvalue>& values)
{
        switch (ftype)
        {
        case mat5_ftype::regular:
                return  read_vector(stream,
                        dtype, dsize, values);

        case mat5_ftype::small:
        default:
                return  read_vector(mem_istream_t(reinterpret_cast<const char*>(&bytes), sizeof(bytes)),
                        dtype, dsize, values);
        }
}

mat5_section_t::mat5_section_t(const mat5_ptype ptype) :
        m_ptype(ptype)
{
}

bool mat5_section_t::load(istream_t& stream)
{
        uint32_t dtype, bytes;
        if (    !stream.read(dtype) ||
                !stream.read(bytes))
        {
                return false;
        }

        // small data format
        if ((dtype >> 16) != 0)
        {
                m_size = 8;
                m_dsize = 4;
                m_dtype = make_dtype((dtype << 16) >> 16);
                m_ftype = mat5_ftype::small;
                m_bytes = bytes;
        }

        // regular format
        else
        {
                const auto compressed = make_dtype(dtype) == mat5_dtype::miCOMPRESSED;
                const auto modulo8 = bytes % 8;
                m_size = compressed ? (8 + bytes) : (8 + bytes + (modulo8 ? 8 - modulo8 : 0));
                m_dsize = m_size - 8;
                m_dtype = make_dtype(dtype);
                m_ftype = mat5_ftype::regular;
        }

        return true;
}

bool mat5_section_t::skip(istream_t& stream) const
{
        switch (m_ftype)
        {
        case mat5_ftype::regular:
                return stream.skip(m_dsize);

        case mat5_ftype::small:
        default:
                return true;
        }
}

bool mat5_section_t::matrix_meta(istream_t& stream) const
{
        switch (m_ptype)
        {
        case mat5_ptype::miMATRIX:
                return skip(stream);

        default:
                return false;
        }
}

bool mat5_section_t::matrix_data(istream_t& stream) const
{
        switch (m_ptype)
        {
        case mat5_ptype::miMATRIX:
                return stream;

        default:
                return false;
        }
}

bool mat5_section_t::matrix_name(istream_t& stream, std::string& name) const
{
        std::vector<char> buffer;
        switch (m_ptype)
        {
        case mat5_ptype::miMATRIX:
                if (!read_vector(stream, m_ftype, m_bytes, m_dtype, m_dsize, buffer))
                {
                        return false;
                }
                name.assign(buffer.data(), buffer.size());
                return true;

        default:
                return false;
        }
}

bool mat5_section_t::matrix_dims(istream_t& stream, std::vector<int32_t>& dims) const
{
        switch (m_ptype)
        {
        case mat5_ptype::miMATRIX:
                return read_vector(stream, m_ftype, m_bytes, m_dtype, m_dsize, dims);

        default:
                return false;
        }
}

std::ostream& nano::operator<<(std::ostream& ostream, const mat5_section_t& sect)
{
        ostream << "type = " << to_string(sect.m_ptype) << "/" << to_string(sect.m_dtype)
                << ", format = " << to_string(sect.m_ftype)
                << ", size = " << sect.m_size << "B"
                << ", data size = " << sect.m_dsize << "B";
        return ostream;
}

static bool load_mat5(istream_t& stream,
        const mat5_section_callback_t& scallback,
        const mat5_error_callback_t& ecallback,
        const mat5_ptype ptype = mat5_ptype::none)
{
        while (stream)
        {
                mat5_section_t section(ptype);
                if (!section.load(stream))
                {
                        ecallback("failed to load section!");
                        return false;
                }

                switch (section.m_dtype)
                {
                case mat5_dtype::miCOMPRESSED:
                        {
                                // gzip compressed section
                                zlib_istream_t zstream(stream, section.m_dsize);
                                if (!load_mat5(zstream, scallback, ecallback))
                                {
                                        return false;
                                }
                        }
                        break;

                case mat5_dtype::miMATRIX:
                        {
                                // array/matrix section, so read the sub-elements
                                if (!load_mat5(stream, scallback, ecallback, mat5_ptype::miMATRIX))
                                {
                                        return false;
                                }
                        }
                        break;

                default:
                        if (!scallback(section, stream))
                        {
                                return false;
                        }
                }
        }

        return true;
}

bool nano::load_mat5(const std::string& path,
        const mat5_header_callback_t& hcallback,
        const mat5_section_callback_t& scallback,
        const mat5_error_callback_t& ecallback)
{
        std::ifstream istream(path.c_str(), std::ios::binary | std::ios::in);
        if (!istream.is_open())
        {
                ecallback("failed to open file <" + path + ">!");
                return false;
        }

        std_istream_t stream(istream);

        // load header
        mat5_header_t header;
        if (!header.load(stream))
        {
                ecallback("failed to load header!");
                return false;
        }
        if (!hcallback(header))
        {
                return false;
        }

        // load sections
        return ::load_mat5(stream, scallback, ecallback);
}
