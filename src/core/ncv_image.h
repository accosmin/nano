#ifndef  NANOCV_IMAGE_H
#define  NANOCV_IMAGE_H

#include "ncv_string.h"
#include "ncv_math.h"

namespace ncv
{
        ////////////////////////////////////////////////////////////////////////////////////////////////////////
        // Image class & utility functions to:
        //      - load/save from/to disk RGBA images
        //      - convert to/from RGBA from/to RGB and Y channels
        ////////////////////////////////////////////////////////////////////////////////////////////////////////

        // Color channels
        enum class channel : int
        {
                red = 0,                // R
                green,                  // G
                blue,                   // B
                luma                    // Y/L
        };

        namespace text
        {
                template <>
                inline string_t to_string(channel dtype)
                {
                        switch (dtype)
                        {
                        case channel::red:              return "red";
                        case channel::green:            return "green";
                        case channel::blue:             return "blue";
                        case channel::luma:             return "luma";
                        default:                        return "luma";
                        }
                }

                template <>
                inline channel from_string<channel>(const string_t& string)
                {
                        if (string == "red")            return channel::red;
                        if (string == "green")          return channel::green;
                        if (string == "blue")           return channel::blue;
                        if (string == "luma")           return channel::luma;
                        throw std::invalid_argument("Invalid channel type <" + string + ">!");
                        return channel::luma;
                }
        }

        // RGBA data
        typedef int32_t                         rgba_t;
        typedef matrix<rgba_t>::matrix_t        rgba_matrix_t;

        // Pixel channel data (e.g. red, green, blue, luma) [0, 255]
        typedef int32_t                         pixel_t;
        typedef matrix<pixel_t>::matrix_t       pixel_matrix_t;

        // Manipulate RGBA color space
        namespace color
        {
                inline rgba_t red(rgba_t rgba)     { return (rgba >> 24) & 0xFF; }
                inline rgba_t green(rgba_t rgba)   { return (rgba >> 16) & 0xFF; }
                inline rgba_t blue(rgba_t rgba)    { return (rgba >>  8) & 0xFF; }
                inline rgba_t alpha(rgba_t rgba)   { return (rgba >>  0) & 0xFF; }

                inline rgba_t luma(rgba_t r, rgba_t g, rgba_t b)
                {
                        return (r * 11 + g * 16 + b * 5) / 32;
                }
                inline rgba_t luma(rgba_t rgba)
                {
                        return luma(red(rgba), green(rgba), blue(rgba));
                }

                inline rgba_t rgba(rgba_t r, rgba_t g, rgba_t b, rgba_t a = 255)
                {
                        return ((r & 0xFF) << 24) | ((g & 0xFF) << 16) | ((b & 0xFF) << 8) | (a & 0xFF);
                }
                inline rgba_t rgba(rgba_t luma)
                {
                        return rgba(luma, luma, luma);
                }

                inline rgba_t min() { return 0; }
                inline rgba_t max() { return 255; }
        }

        // Image
        class image
        {
        public:

                // Constructors
                image(size_t rows = 0, size_t cols = 0, const string_t& name = "");
                image(const rgba_matrix_t& rgba, const string_t& name = "");

                // Load image from file or memory
                bool load(const string_t& path);
                bool load(const rgba_matrix_t& rgba);                

                template <typename tchannel>
                bool load(const typename matrix<tchannel>::matrix_t& chd, channel che)
                {
                        return _load<tchannel>(chd, che);
                }

                template <typename tchannel>
                bool load(const typename matrix<tchannel>::matrix_t& chr,
                          const typename matrix<tchannel>::matrix_t& chg,
                          const typename matrix<tchannel>::matrix_t& chb)
                {
                        return _load<tchannel>(chr, chg, chb);
                }

                // Save image to file or memory
                bool save(const string_t& path) const;
                bool save(const string_t& path, channel ch) const;

                template <typename tchannel>
                bool save(typename matrix<tchannel>::matrix_t& chd, channel che) const
                {
                        return _save<tchannel>(chd, che);
                }

                // Rename image
                void rename(const string_t& name) { m_name = name; }

                // Access functions
                size_t rows() const { return math::cast<size_t>(m_rgba.rows()); }
                size_t cols() const { return math::cast<size_t>(m_rgba.cols()); }
                size_t size() const { return math::cast<size_t>(m_rgba.size()); }
                bool empty() const { return size() == 0; }
                const string_t& name() const { return m_name; }

        private:

                //-------------------------------------------------------------------------------------------------

                template <typename tchannel>
                bool _load(const typename matrix<tchannel>::matrix_t& chd, channel che)
                {
                        const size_t rows = math::cast<size_t>(chd.rows());
                        const size_t cols = math::cast<size_t>(chd.cols());
                        const size_t size = rows * cols;

                        m_rgba.resize(rows, cols);

                        switch (che)
                        {
                        case channel::red:
                                for (size_t i = 0; i < size; i ++)
                                {
                                        m_rgba(i) = color::rgba(math::cast<rgba_t>(chd(i)), 0, 0);
                                }
                                break;

                        case channel::green:
                                for (size_t i = 0; i < size; i ++)
                                {
                                        m_rgba(i) = color::rgba(0, math::cast<rgba_t>(chd(i)), 0);
                                }
                                break;

                        case channel::blue:
                                for (size_t i = 0; i < size; i ++)
                                {
                                        m_rgba(i) = color::rgba(0, 0, math::cast<rgba_t>(chd(i)));
                                }
                                break;

                        case channel::luma:
                                for (size_t i = 0; i < size; i ++)
                                {
                                        const rgba_t val = math::cast<rgba_t>(chd(i));
                                        m_rgba(i) = color::rgba(val, val, val);
                                }
                                break;
                        }

                        return true;
                }

                //-------------------------------------------------------------------------------------------------

                template <typename tchannel>
                bool _load(
                        const typename matrix<tchannel>::matrix_t& chr,
                        const typename matrix<tchannel>::matrix_t& chg,
                        const typename matrix<tchannel>::matrix_t& chb)
                {
                        const size_t rows = math::cast<size_t>(chr.rows());
                        const size_t cols = math::cast<size_t>(chr.cols());
                        const size_t size = rows * cols;

                        if (    rows != math::cast<size_t>(chg.rows()) ||
                                rows != math::cast<size_t>(chb.rows()) ||
                                cols != math::cast<size_t>(chg.cols()) ||
                                cols != math::cast<size_t>(chb.cols()))
                        {
                                return false;
                        }

                        m_rgba.resize(rows, cols);

                        for (size_t i = 0; i < size; i ++)
                        {
                                m_rgba(i) = color::rgba(math::cast<rgba_t>(chr(i)),
                                                        math::cast<rgba_t>(chg(i)),
                                                        math::cast<rgba_t>(chb(i)));
                        }

                        return true;
                }

                //-------------------------------------------------------------------------------------------------

                template <typename tchannel>
                bool _save(typename matrix<tchannel>::matrix_t& chd, channel che) const
                {
                        chd.resize(rows(), cols());

                        const size_t size = this->size();

                        switch (che)
                        {
                        case channel::red:
                                for (size_t i = 0; i < size; i ++)
                                {
                                        chd(i) = math::cast<tchannel>(color::red(m_rgba(i)));
                                }
                                break;

                        case channel::green:
                                for (size_t i = 0; i < size; i ++)
                                {
                                        chd(i) = math::cast<tchannel>(color::green(m_rgba(i)));
                                }
                                break;

                        case channel::blue:
                                for (size_t i = 0; i < size; i ++)
                                {
                                        chd(i) = math::cast<tchannel>(color::blue(m_rgba(i)));
                                }
                                break;

                        case channel::luma:
                                for (size_t i = 0; i < size; i ++)
                                {
                                        chd(i) = math::cast<tchannel>(color::luma(m_rgba(i)));
                                }
                                break;
                        }

                        return true;
                }

                //-------------------------------------------------------------------------------------------------

        private:

                // Attributes
                rgba_matrix_t                   m_rgba;
                string_t                        m_name;
        };
}

#endif //  NANOCV_IMAGE_H
