#pragma once

#include "activation.h"

namespace nano
{
        ///
        /// \brief identity activation function.
        ///
        struct unit_activation_t
        {
                template <typename tiarray>
                static auto output(const tiarray& idata)
                {
                        return idata;
                }

                template <typename tiarray>
                static auto ginput(const tiarray&)
                {
                        return 1;
                }
        };

        using unit_activation_layer_t = activation_layer_t<unit_activation_t>;
}
