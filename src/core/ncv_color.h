#ifndef  NANOCV_COLOR_H
#define  NANOCV_COLOR_H

#include "ncv_string.h"
#include "ncv_math.h"

namespace ncv
{
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

        // RGBA
        typedef uint32_t                        rgba_t;
        typedef matrix<rgba_t>::matrix_t        rgba_matrix_t;

        // CIELab
        typedef Eigen::Vector3d                 cielab_t;
        typedef matrix<cielab_t>::matrix_t      cielab_matrix_t;

        // manipulate color space
        namespace color
        {
                // RGBA decoding (R, G, B, A, L(uma), CIELab)
                inline rgba_t decode_red(rgba_t rgba)     { return (rgba >> 24) & 0xFF; }
                inline rgba_t decode_green(rgba_t rgba)   { return (rgba >> 16) & 0xFF; }
                inline rgba_t decode_blue(rgba_t rgba)    { return (rgba >>  8) & 0xFF; }
                inline rgba_t decode_alpha(rgba_t rgba)   { return (rgba >>  0) & 0xFF; }

                inline rgba_t decode_luma(rgba_t r, rgba_t g, rgba_t b)
                {
                        return (r * 11 + g * 16 + b * 5) / 32;
                }
                inline rgba_t decode_luma(rgba_t rgba)
                {
                        return decode_luma(decode_red(rgba),
                                           decode_green(rgba),
                                           decode_blue(rgba));
                }

                cielab_t decode_cielab(rgba_t rgba);

                // RGBA encoding (R, G, B, A, CIELab)
                inline rgba_t encode_rgba(rgba_t r, rgba_t g, rgba_t b, rgba_t a = 255)
                {
                        return ((r & 0xFF) << 24) | ((g & 0xFF) << 16) | ((b & 0xFF) << 8) | (a & 0xFF);
                }

                rgba_t encode_cielab(const cielab_t& cielab);

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
        }
}

#endif //  NANOCV_COLOR_H
