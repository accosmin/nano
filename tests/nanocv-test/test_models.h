#pragma once

#include "libnanocv/string.h"
#include "libnanocv/vision/color.h"

namespace test
{
        ///
        /// \brief create the <model description, loss id> configurations to test gradients with
        ///
        NANOCV_PUBLIC std::vector<std::pair<ncv::string_t, ncv::string_t> > make_grad_configs(
                ncv::size_t& irows, ncv::size_t& icols, ncv::size_t& outputs, ncv::color_mode& color);
}
