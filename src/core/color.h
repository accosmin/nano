#ifndef  NANOCV_COLOR_H
#define  NANOCV_COLOR_H

#include "types.h"

namespace ncv
{
        // RGBA
        typedef uint32_t                                                        rgba_t;
        typedef tensor::matrix_types_t<rgba_t>::matrix_t                        rgba_matrix_t;
        typedef std::vector<rgba_t>                                             rgbas_t;

        // CIELab
        typedef tensor::fixed_size_vector_types_t<scalar_t, 3>::vector_t        cielab_t;
        typedef tensor::matrix_types_t<cielab_t>::matrix_t                      cielab_matrix_t;

        // manipulate colors
        namespace color
        {
                // RGBA decoding (R, G, B, A, L(uma), CIELab)
                inline rgba_t make_red(rgba_t rgba)     { return (rgba >> 24) & 0xFF; }
                inline rgba_t make_green(rgba_t rgba)   { return (rgba >> 16) & 0xFF; }
                inline rgba_t make_blue(rgba_t rgba)    { return (rgba >>  8) & 0xFF; }
                inline rgba_t make_alpha(rgba_t rgba)   { return (rgba >>  0) & 0xFF; }

                inline rgba_t make_luma_(rgba_t r, rgba_t g, rgba_t b)
                {
                        return (r * 11 + g * 16 + b * 5) / 32;
                }
                inline rgba_t make_luma(rgba_t rgba)
                {
                        return make_luma_(make_red(rgba), make_green(rgba), make_blue(rgba));
                }

                cielab_t make_cielab(rgba_t rgba);

                // RGBA encoding (R, G, B, A, CIELab)
                inline rgba_t make_rgba(rgba_t r, rgba_t g, rgba_t b, rgba_t a = 255)
                {
                        return ((r & 0xFF) << 24) | ((g & 0xFF) << 16) | ((b & 0xFF) << 8) | (a & 0xFF);
                }

                rgba_t make_rgba(const cielab_t& cielab);

                // RGBA encoding (by normalizing a matrix data)
                rgba_matrix_t make_rgba(const matrix_t& data);
                rgba_matrix_t make_rgba(const matrix_t& data, scalar_t min, scalar_t max);

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
