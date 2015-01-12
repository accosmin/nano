#pragma once

#include "types.h"

namespace ncv
{
        // RGBA
        typedef uint32_t                                                        rgba_t;
        typedef tensor::matrix_types_t<rgba_t>::tmatrix                         rgba_matrix_t;

        // CIELab
        typedef tensor::fixed_size_vector_types_t<scalar_t, 3>::tvector         cielab_t;
        typedef tensor::matrix_types_t<cielab_t>::tmatrix                       cielab_matrix_t;

        // Grayscale
        typedef uint8_t                                                         luma_t;
        typedef tensor::matrix_types_t<luma_t>::tmatrix                         luma_matrix_t;

        // manipulate colors
        namespace color
        {
                // RGBA decoding (R, G, B, A, L(uma), CIELab)
                inline rgba_t make_red(rgba_t rgba)     { return (rgba >> 24) & 0xFF; }
                inline rgba_t make_green(rgba_t rgba)   { return (rgba >> 16) & 0xFF; }
                inline rgba_t make_blue(rgba_t rgba)    { return (rgba >>  8) & 0xFF; }
                inline rgba_t make_alpha(rgba_t rgba)   { return (rgba >>  0) & 0xFF; }
                inline rgba_t make_opaque(rgba_t rgba)  { return rgba | 0xFF; }

                inline rgba_t make_luma(rgba_t r, rgba_t g, rgba_t b)
                {
                        return (r * 11 + g * 16 + b * 5) / 32;
                }
                inline rgba_t make_luma(rgba_t rgba)
                {
                        return make_luma(make_red(rgba), make_green(rgba), make_blue(rgba));
                }

                cielab_t make_cielab(rgba_t rgba);

                // RGBA encoding (R, G, B, A, CIELab)
                inline rgba_t make_rgba(rgba_t r, rgba_t g, rgba_t b, rgba_t a = 255)
                {
                        return ((r & 0xFF) << 24) | ((g & 0xFF) << 16) | ((b & 0xFF) << 8) | (a & 0xFF);
                }                
                inline rgba_t make_rgba(luma_t l, rgba_t a = 255)
                {
                        return make_rgba(l, l, l, a);
                }

                rgba_t make_rgba(const cielab_t& cielab);

                // color channel range
                inline scalar_t min(color_channel ch)
                {
                        switch (ch)
                        {
                        case color_channel::red:        return 0.0;
                        case color_channel::green:      return 0.0;
                        case color_channel::blue:       return 0.0;
                        case color_channel::luma:       return 0.0;
                        case color_channel::cielab_l:   return 0.0;
                        case color_channel::cielab_a:   return -86.1846;
                        case color_channel::cielab_b:   return -107.864;
                        default:                        return 0.0;
                        }
                }

                inline scalar_t max(color_channel ch)
                {
                        switch (ch)
                        {
                        case color_channel::red:        return 255.0;
                        case color_channel::green:      return 255.0;
                        case color_channel::blue:       return 255.0;
                        case color_channel::luma:       return 255.0;
                        case color_channel::cielab_l:   return 100.0;
                        case color_channel::cielab_a:   return 98.2542;
                        case color_channel::cielab_b:   return 94.4825;
                        default:                        return 255.0;
                        }
                }
        }
}

