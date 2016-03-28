#pragma once

#include "arch.h"
#include "tensor.h"
#include "text/enum_string.hpp"
#include <iosfwd>
#include <cstdint>

namespace nano
{
        using luma_t = uint8_t;
        using rgba_t = Eigen::Matrix<uint8_t, 4, 1>;

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

        NANO_PUBLIC std::ostream& operator<<(std::ostream&, const color_mode);

        ///
        /// \brief transform RGB to luma
        ///
        template
        <
                typename tred,
                typename tgreen,
                typename tblue
        >
        auto make_luma(const tred& r, const tgreen& g, const tblue& b)
        {
                return (r * 11 + g * 16 + b * 5) / 32;
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

