#pragma once

#include <cstdint>

namespace zob
{
        ///
        /// \brief image coordinate (in pixels) - 64bit to be compatible with Eigen::Index
        ///
        using coord_t = int64_t;

        ///
        /// \brief image area (in pixels)
        /// 
        using area_t = int64_t;
}


