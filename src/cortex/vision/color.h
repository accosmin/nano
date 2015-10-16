#pragma once

#include "arch.h"
#include "math/cast.hpp"
#include "math/clamp.hpp"
#include "cortex/tensor.h"
#include "text/enum_string.hpp"
#include <iosfwd>
#include <cstdint>

namespace ncv
{
        // RGBA
        typedef uint32_t                                        rgba_t;
        typedef tensor::matrix_t<rgba_t>                        rgba_matrix_t;

        // CIELab
        typedef tensor::fixed_size_vector_t<scalar_t, 4>        cielab_t;
        typedef tensor::matrix_t<cielab_t>                      cielab_matrix_t;

        // grayscale
        typedef uint8_t                                         luma_t;
        typedef tensor::matrix_t<luma_t>                        luma_matrix_t;

        ///
        /// \brief color channels
        ///
        enum class color_channel
        {
                red = 0,                // R
                green,                  // G
                blue,                   // B
                luma,                   // Y/L
                rgba,                   // RGBA
                alpha,                  // transparency
                cielab_l,               // CIELab L
                cielab_a,               // CIELab a
                cielab_b                // CIELab b
        };

        ///
        /// \brief color processing mode methods
        ///
        enum class color_mode
        {
                luma,                   ///< process only grayscale color channel
                rgba                    ///< process red, green & blue color channels
        };

        NANOCV_PUBLIC std::ostream& operator<<(std::ostream&, color_mode);

        // manipulate colors
        namespace color
        {
                // RGBA decoding (R, G, B, A, L(uma), CIELab)
                inline rgba_t get_red(rgba_t rgba)                      { return (rgba >> 24) & 0xFF; }
                inline rgba_t get_green(rgba_t rgba)                    { return (rgba >> 16) & 0xFF; }
                inline rgba_t get_blue(rgba_t rgba)                     { return (rgba >>  8) & 0xFF; }
                inline rgba_t get_alpha(rgba_t rgba)                    { return (rgba >>  0) & 0xFF; }

                inline rgba_t set_red(rgba_t rgba, rgba_t v)            { return (rgba & 0x00FFFFFF) | (v << 24); }
                inline rgba_t set_green(rgba_t rgba, rgba_t v)          { return (rgba & 0xFF00FFFF) | (v << 16); }
                inline rgba_t set_blue(rgba_t rgba, rgba_t v)           { return (rgba & 0xFFFF00FF) | (v <<  8); }
                inline rgba_t set_alpha(rgba_t rgba, rgba_t v)          { return (rgba & 0xFFFFFF00) | (v <<  0); }

                inline luma_t make_luma(rgba_t r, rgba_t g, rgba_t b)
                {
                        return math::cast<luma_t>((r * 11 + g * 16 + b * 5) / 32);
                }
                inline luma_t make_luma(rgba_t rgba)
                {
                        return make_luma(get_red(rgba), get_green(rgba), get_blue(rgba));
                }

                NANOCV_PUBLIC cielab_t make_cielab(rgba_t rgba);

                // RGBA encoding (R, G, B, A, CIELab)
                inline rgba_t make_rgba(rgba_t r, rgba_t g, rgba_t b, rgba_t a = 255)
                {
                        return (r << 24) | (g << 16) | (b << 8) | a;
                }                
                inline rgba_t make_rgba(luma_t l, rgba_t a = 255)
                {
                        return make_rgba(l, l, l, a);
                }

                NANOCV_PUBLIC rgba_t make_rgba(const cielab_t& cielab);

                ///
                /// \brief minimum color range
                ///
                NANOCV_PUBLIC scalar_t min(color_channel ch);

                ///
                /// \brief maximum color range
                ///
                NANOCV_PUBLIC scalar_t max(color_channel ch);

                ///
                /// \brief create random RGBA color
                ///
                NANOCV_PUBLIC rgba_t make_random_rgba();

                ///
                /// \brief create random RGBA color as opposite as possible from the source color
                ///
                NANOCV_PUBLIC rgba_t make_opposite_random_rgba(const rgba_t source);

                ///
                /// \brief transform patch to scaled [0, 1] tensor with 1 plane (luma)
                ///
                NANOCV_PUBLIC tensor_t to_luma_tensor(const luma_matrix_t& patch);

                ///
                /// \brief transform patch to scaled [0, 1] tensor with 3 planes (rgb)
                ///
                NANOCV_PUBLIC tensor_t to_rgb_tensor(const rgba_matrix_t& patch);

                ///
                /// \brief transform patch to scaled [0, 1] tensor with 4 planes (rgba)
                ///
                NANOCV_PUBLIC tensor_t to_rgba_tensor(const rgba_matrix_t& patch);

                ///
                /// \brief transform 1 plane scaled [0, 1] patch to luma matrix
                ///
                NANOCV_PUBLIC luma_matrix_t from_luma_tensor(const tensor_t& patch);

                ///
                /// \brief transform 3 planes scaled [0, 1] patch to rgb matrix
                ///
                NANOCV_PUBLIC rgba_matrix_t from_rgb_tensor(const tensor_t& patch);

                ///
                /// \brief transform 4 planes scaled [0, 1] patch to rgba matrix
                ///
                NANOCV_PUBLIC rgba_matrix_t from_rgba_tensor(const tensor_t& patch);
        }        
}

namespace text
{
        template <>
        inline std::map<ncv::color_mode, std::string> enum_string<ncv::color_mode>()
        {
                return
                {
                        { ncv::color_mode::luma, "luma" },
                        { ncv::color_mode::rgba, "rgba" }
                };
        }

        template <>
        inline std::map<ncv::color_channel, std::string> enum_string<ncv::color_channel>()
        {
                return
                {
                        { ncv::color_channel::red,           "red" },
                        { ncv::color_channel::green,         "green" },
                        { ncv::color_channel::blue,          "blue" },
                        { ncv::color_channel::luma,          "luma" },
                        { ncv::color_channel::rgba,          "rgba" },
                        { ncv::color_channel::alpha,         "alpha" },
                        { ncv::color_channel::cielab_l,      "cielab_l" },
                        { ncv::color_channel::cielab_a,      "cielab_a" },
                        { ncv::color_channel::cielab_b,      "cielab_b" }
                };
        }
}

