#ifndef  NANOCV_IMAGE_H
#define  NANOCV_IMAGE_H

#include "ncv_string.h"
#include "ncv_math.h"

namespace ncv
{
        ////////////////////////////////////////////////////////////////////////////////////////////////////////
        // image class & utility functions to:
        //      - load/save from/to disk RGBA images
        //      - convert to/from RGBA from/to RGB and Y channels
        ////////////////////////////////////////////////////////////////////////////////////////////////////////

        // color channels
        enum class channel : int
        {
                red = 0,                // R
                green,                  // G
                blue,                   // B
                luma,                   // Y/L
                cielab_l,               // CIELab L
                cielab_a,               // CIELab a
                cielab_b                // CIELab b
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
                        case channel::cielab_l:         return "cielab_l";
                        case channel::cielab_a:         return "cielab_a";
                        case channel::cielab_b:         return "cielab_b";
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
                        if (string == "cielab_l")       return channel::cielab_l;
                        if (string == "cielab_a")       return channel::cielab_a;
                        if (string == "cielab_b")       return channel::cielab_b;
                        throw std::invalid_argument("Invalid channel type <" + string + ">!");
                        return channel::luma;
                }
        }

        // RGBA data
        typedef int32_t                         rgba_t;
        typedef int32_t                         rgb_t;
        typedef matrix<rgba_t>::matrix_t        rgba_matrix_t;

        // pixel channel data (e.g. red, green, blue, luma) [0, 255]
        typedef int32_t                         pixel_t;
        typedef matrix<pixel_t>::matrix_t       pixel_matrix_t; // FIXME: TO BE REMOVED!

        // manipulate color space
        namespace color
        {
                // RGBA transform
                inline rgba_t rgba2r(rgba_t rgba)       { return (rgba >> 24) & 0xFF; }
                inline rgba_t rgba2g(rgba_t rgba)       { return (rgba >> 16) & 0xFF; }
                inline rgba_t rgba2b(rgba_t rgba)       { return (rgba >>  8) & 0xFF; }
                inline rgba_t rgba2a(rgba_t rgba)       { return (rgba >>  0) & 0xFF; }

                inline rgba_t rgba2l(rgba_t r, rgba_t g, rgba_t b)
                {
                        return (r * 11 + g * 16 + b * 5) / 32;
                }
                inline rgba_t rgba2l(rgba_t rgba)
                {
                        return rgba2l(rgba2r(rgba), rgba2g(rgba), rgba2b(rgba));
                }

                inline rgba_t make_rgba(rgba_t r, rgba_t g, rgba_t b, rgba_t a = 255)
                {
                        return ((r & 0xFF) << 24) | ((g & 0xFF) << 16) | ((b & 0xFF) << 8) | (a & 0xFF);
                }
                inline rgba_t make_rgba(rgba_t luma)
                {
                        return make_rgba(luma, luma, luma);
                }

                // CIELab transform
                void rgb2lab(rgb_t rgb_r, rgb_t rgb_g, rgb_t rgb_b, scalar_t& cie_l, scalar_t& cie_a, scalar_t& cie_b);
                void lab2rgb(scalar_t cie_l, scalar_t cie_a, scalar_t cie_b, rgb_t& rgb_r, rgb_t& rgb_g, rgb_t& rgb_b);

                // color channel range
                inline scalar_t min(channel ch)
                {
                        switch (ch)
                        {
                        case channel::red:      return 0.0;
                        case channel::green:    return 0.0;
                        case channel::blue:     return 0.0;
                        case channel::luma:     return 0.0;
                        case channel::cielab_l: return 0.0;
                        case channel::cielab_a: return -86.1846;
                        case channel::cielab_b: return -107.864;
                        default:                return 0.0;
                        }
                }

                inline scalar_t max(channel ch)
                {
                        switch (ch)
                        {
                        case channel::red:      return 255.0;
                        case channel::green:    return 255.0;
                        case channel::blue:     return 255.0;
                        case channel::luma:     return 255.0;
                        case channel::cielab_l: return 100.0;
                        case channel::cielab_a: return 98.2542;
                        case channel::cielab_b: return 94.4825;
                        default:                return 255.0;
                        }
                }

                // TODO: functions to return an RGBA encoder/decoder based on the channel type
                //      to simplify image::load/save!!!
        }

        // image
        class image
        {
        public:

                // constructors
                image(size_t rows = 0, size_t cols = 0, const string_t& name = "");
                image(const rgba_matrix_t& rgba, const string_t& name = "");

                // load image from file or memory
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

                // save image to file or memory
                bool save(const string_t& path) const;
                bool save(const string_t& path, channel ch) const;

                template <typename tchannel>
                bool save(typename matrix<tchannel>::matrix_t& chd, channel che) const
                {
                        return _save<tchannel>(chd, che);
                }

                // rename image
                void rename(const string_t& name) { m_name = name; }

                // access functions
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
                                        m_rgba(i) = color::make_rgba(math::cast<rgba_t>(chd(i)), 0, 0);
                                }
                                break;

                        case channel::green:
                                for (size_t i = 0; i < size; i ++)
                                {
                                        m_rgba(i) = color::make_rgba(0, math::cast<rgba_t>(chd(i)), 0);
                                }
                                break;

                        case channel::blue:
                                for (size_t i = 0; i < size; i ++)
                                {
                                        m_rgba(i) = color::make_rgba(0, 0, math::cast<rgba_t>(chd(i)));
                                }
                                break;

                        case channel::luma:
                                for (size_t i = 0; i < size; i ++)
                                {
                                        const rgba_t val = math::cast<rgba_t>(chd(i));
                                        m_rgba(i) = color::make_rgba(val, val, val);
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
                                m_rgba(i) = color::make_rgba(math::cast<rgba_t>(chr(i)),
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
                                        chd(i) = math::cast<tchannel>(color::rgba2r(m_rgba(i)));
                                }
                                break;

                        case channel::green:
                                for (size_t i = 0; i < size; i ++)
                                {
                                        chd(i) = math::cast<tchannel>(color::rgba2g(m_rgba(i)));
                                }
                                break;

                        case channel::blue:
                                for (size_t i = 0; i < size; i ++)
                                {
                                        chd(i) = math::cast<tchannel>(color::rgba2b(m_rgba(i)));
                                }
                                break;

                        case channel::luma:
                                for (size_t i = 0; i < size; i ++)
                                {
                                        chd(i) = math::cast<tchannel>(color::rgba2l(m_rgba(i)));
                                }
                                break;
                        }

                        return true;
                }

                //-------------------------------------------------------------------------------------------------

        private:

                // attributes
                rgba_matrix_t                   m_rgba;
                string_t                        m_name;
        };
}

#endif //  NANOCV_IMAGE_H
