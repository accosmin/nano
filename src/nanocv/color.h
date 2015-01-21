#pragma once

#include "types.h"
#include "util/cast.hpp"
#include "util/math.hpp"

namespace ncv
{
        // RGBA
        typedef uint32_t                                                        rgba_t;
        typedef tensor::matrix_types_t<rgba_t>::tmatrix                         rgba_matrix_t;

        // CIELab
        typedef tensor::fixed_size_vector_types_t<scalar_t, 4>::tvector         cielab_t;
        typedef tensor::matrix_types_t<cielab_t>::tmatrix                       cielab_matrix_t;

        // grayscale
        typedef uint8_t                                                         luma_t;
        typedef tensor::matrix_types_t<luma_t>::tmatrix                         luma_matrix_t;

        // manipulate colors
        namespace color
        {
                // RGBA decoding (R, G, B, A, L(uma), CIELab)
                inline rgba_t get_red(rgba_t rgba)                      { return (rgba >> 24) & 0xFF; }
                inline rgba_t get_green(rgba_t rgba)                    { return (rgba >> 16) & 0xFF; }
                inline rgba_t get_blue(rgba_t rgba)                     { return (rgba >>  8) & 0xFF; }
                inline rgba_t get_alpha(rgba_t rgba)                    { return (rgba >>  0) & 0xFF; }
                inline luma_t get_luma(luma_t luma)                     { return luma; }

                inline rgba_t set_red(rgba_t rgba, rgba_t v)            { return (rgba & 0x00FFFFFF) | (v << 24); }
                inline rgba_t set_green(rgba_t rgba, rgba_t v)          { return (rgba & 0xFF00FFFF) | (v << 16); }
                inline rgba_t set_blue(rgba_t rgba, rgba_t v)           { return (rgba & 0xFFFF00FF) | (v <<  8); }
                inline rgba_t set_alpha(rgba_t rgba, rgba_t v)          { return (rgba & 0xFFFFFF00) | (v <<  0); }
                inline luma_t set_luma(luma_t, luma_t v)                { return v; }

                inline luma_t make_luma(rgba_t r, rgba_t g, rgba_t b)
                {
                        return static_cast<luma_t>((r * 11 + g * 16 + b * 5) / 32);
                }
                inline luma_t make_luma(rgba_t rgba)
                {
                        return make_luma(get_red(rgba), get_green(rgba), get_blue(rgba));
                }

                cielab_t make_cielab(rgba_t rgba);

                // RGBA encoding (R, G, B, A, CIELab)
                inline rgba_t make_rgba(rgba_t r, rgba_t g, rgba_t b, rgba_t a = 255)
                {
                        return (r << 24) | (g << 16) | (b << 8) | a;
                }                
                inline rgba_t make_rgba(luma_t l, rgba_t a = 255)
                {
                        return make_rgba(l, l, l, a);
                }

                rgba_t make_rgba(const cielab_t& cielab);

                // interpolate luma
                inline luma_t luma_mixer(
                        scalar_t w0, luma_t l0,
                        scalar_t w1, luma_t l1,
                        scalar_t w2, luma_t l2,
                        scalar_t w3, luma_t l3)
                {
                        const scalar_t l = w0 * l0 + w1 * l1 + w2 * l2 + w3 * l3;

                        return math::cast<luma_t>(math::clamp(l, scalar_t(0), scalar_t(255)));
                }

                // interpolate rgba
                inline rgba_t rgba_mixer(
                        scalar_t w0, rgba_t c0,
                        scalar_t w1, rgba_t c1,
                        scalar_t w2, rgba_t c2,
                        scalar_t w3, rgba_t c3)
                {
                        const scalar_t r = w0 * get_red(c0) + w1 * get_red(c1) + w2 * get_red(c2) + w3 * get_red(c3);
                        const scalar_t g = w0 * get_green(c0) + w1 * get_green(c1) + w2 * get_green(c2) + w3 * get_green(c3);
                        const scalar_t b = w0 * get_blue(c0) + w1 * get_blue(c1) + w2 * get_blue(c2) + w3 * get_blue(c3);
                        const scalar_t a = w0 * get_alpha(c0) + w1 * get_alpha(c1) + w2 * get_alpha(c2) + w3 * get_alpha(c3);

                        return  make_rgba(
                                math::cast<rgba_t>(math::clamp(r, scalar_t(0), scalar_t(255))),
                                math::cast<rgba_t>(math::clamp(g, scalar_t(0), scalar_t(255))),
                                math::cast<rgba_t>(math::clamp(b, scalar_t(0), scalar_t(255))),
                                math::cast<rgba_t>(math::clamp(a, scalar_t(0), scalar_t(255))));

                }

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

