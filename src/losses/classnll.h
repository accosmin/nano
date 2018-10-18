#pragma once

#include "classification.h"

namespace nano
{
        ///
        /// \brief class negative log-likelihood loss (also called cross-entropy loss).
        ///
        struct classnll_t
        {
                template <typename tarray>
                static auto value(const tarray& target, const tarray& output)
                {
                        return  std::log(output.exp().sum()) -
                                scalar_t(0.5) * ((1 + target) * output).sum();
                }

                template <typename tarray>
                static auto vgrad(const tarray& target, const tarray& output)
                {
                        return  output.exp() / (output.exp().sum()) -
                                scalar_t(0.5) * (1 + target);
                }
        };

        using classnll_loss_t = sclassification_t<classnll_t>;
}
