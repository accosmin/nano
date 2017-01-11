#pragma once

#include "arch.h"
#include "tensor.h"
#include "text/to_string.h"
#include <ostream>
#include <cstdint>

namespace nano
{
        using luma_t = uint8_t;
        using rgb_t  = tensor::vector_t<uint8_t, 3>;
        using rgba_t = tensor::vector_t<uint8_t, 4>;

        /// 3D buffer to store image patches (number of dimensions is equal to color channels)
        using image_tensor_t = tensor::tensor_t<luma_t, 3>;

        ///
        /// \brief color storage & processing modes
        ///
        enum class color_mode
        {
                luma,           ///< luma/grayscale (1 band)
                rgba,           ///< RGBA (4 bands)
                rgb             ///< RGB (3 bands)
        };

        inline std::ostream& operator<<(std::ostream& os, const color_mode mode)
        {
                return os << to_string(mode);
        }

        ///
        /// \brief transform RGB to luma
        ///
        template
        <
                typename tmatrix
        >
        auto make_luma(const tmatrix& r, const tmatrix& g, const tmatrix& b)
        {
                return (r.template cast<uint32_t>() * 11 +
                        g.template cast<uint32_t>() * 16 +
                        b.template cast<uint32_t>() * 5) / 32;
        }
        template <>
        inline auto make_luma<luma_t>(const luma_t& r, const luma_t& g, const luma_t& b)
        {
                return (static_cast<uint32_t>(r) * 11 +
                        static_cast<uint32_t>(g) * 16 +
                        static_cast<uint32_t>(b) * 5) / 32;
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
                        { nano::color_mode::rgba, "rgba" },
                        { nano::color_mode::rgb, "rgb" }
                };
        }
}

