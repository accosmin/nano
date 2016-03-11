#pragma once

#include "arch.h"
#include "tensor.h"
#include "math/cast.hpp"
#include "math/clamp.hpp"
#include "text/enum_string.hpp"
#include <iosfwd>
#include <cstdint>

namespace nano
{
        /// RGBA
        using rgba_t = uint32_t;
        using rgba_matrix_t = tensor::matrix_t<rgba_t>;

        /// grayscale
        using luma_t = uint8_t;
        using luma_matrix_t = tensor::matrix_t<luma_t>;

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
        };

        ///
        /// \brief color processing mode methods
        ///
        enum class color_mode
        {
                luma,                   ///< process grayscale color channel
                rgba                    ///< process red, green & blue color channels
        };

        NANO_PUBLIC std::ostream& operator<<(std::ostream&, color_mode);

        // manipulate colors
        namespace color
        {
                // RGBA decoding (R, G, B, A, L(uma))
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
                        return nano::cast<luma_t>((r * 11 + g * 16 + b * 5) / 32);
                }
                inline luma_t make_luma(rgba_t rgba)
                {
                        return make_luma(get_red(rgba), get_green(rgba), get_blue(rgba));
                }

                // RGBA encoding (R, G, B, A)
                inline rgba_t make_rgba(rgba_t r, rgba_t g, rgba_t b, rgba_t a = 255)
                {
                        return (r << 24) | (g << 16) | (b << 8) | a;
                }
                inline rgba_t make_rgba(luma_t l, rgba_t a = 255)
                {
                        return make_rgba(l, l, l, a);
                }

                ///
                /// \brief minimum color range
                ///
                NANO_PUBLIC scalar_t min(color_channel ch);

                ///
                /// \brief maximum color range
                ///
                NANO_PUBLIC scalar_t max(color_channel ch);

                ///
                /// \brief create random RGBA color
                ///
                NANO_PUBLIC rgba_t make_random_rgba();

                ///
                /// \brief create random RGBA color as opposite as possible from the source color
                ///
                NANO_PUBLIC rgba_t make_opposite_random_rgba(const rgba_t source);

                ///
                /// \brief transform patch to scaled [0, 1] tensor with 1 plane (luma)
                ///
                NANO_PUBLIC tensor_t to_luma_tensor(const luma_matrix_t& patch);

                ///
                /// \brief transform patch to scaled [0, 1] tensor with 3 planes (rgb)
                ///
                NANO_PUBLIC tensor_t to_rgb_tensor(const rgba_matrix_t& patch);

                ///
                /// \brief transform patch to scaled [0, 1] tensor with 4 planes (rgba)
                ///
                NANO_PUBLIC tensor_t to_rgba_tensor(const rgba_matrix_t& patch);

                ///
                /// \brief transform 1 plane scaled [0, 1] patch to luma matrix
                ///
                NANO_PUBLIC luma_matrix_t from_luma_tensor(const tensor_t& patch);

                ///
                /// \brief transform 3 planes scaled [0, 1] patch to rgb matrix
                ///
                NANO_PUBLIC rgba_matrix_t from_rgb_tensor(const tensor_t& patch);

                ///
                /// \brief transform 4 planes scaled [0, 1] patch to rgba matrix
                ///
                NANO_PUBLIC rgba_matrix_t from_rgba_tensor(const tensor_t& patch);
        }
}

namespace nano
{
        template <>
        inline std::map<nano::color_mode, std::string> enum_string<nano::color_mode>()
        {
                return
                {
                        { nano::color_mode::luma, "luma" },
                        { nano::color_mode::rgba, "rgba" }
                };
        }

        template <>
        inline std::map<nano::color_channel, std::string> enum_string<nano::color_channel>()
        {
                return
                {
                        { nano::color_channel::red,           "red" },
                        { nano::color_channel::green,         "green" },
                        { nano::color_channel::blue,          "blue" },
                        { nano::color_channel::luma,          "luma" },
                        { nano::color_channel::rgba,          "rgba" },
                        { nano::color_channel::alpha,         "alpha" }
                };
        }
}

