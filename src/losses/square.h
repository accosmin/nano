#pragma once

#include "regression.h"
#include "classification.h"

namespace nano
{
        ///
        /// \brief square loss: l(x) = 1/2 * x^x.
        ///
        /// usage:
        ///     - regression:           l(output, target) = l(output - target)
        ///     - classification:       l(output, target) = l(1 - output * target)
        ///
        struct square_regression_t
        {
                template <typename tarray>
                static auto value(const tarray& target, const tarray& output)
                {
                        return scalar_t(0.5) * (output - target).square().sum();
                }

                template <typename tarray>
                static auto vgrad(const tarray& target, const tarray& output)
                {
                        return output - target;
                }
        };

        struct square_classification_t
        {
                template <typename tarray>
                static auto value(const tarray& target, const tarray& output)
                {
                        return scalar_t(0.5) * (1 - output * target).square().sum();
                }

                template <typename tarray>
                static auto vgrad(const tarray& target, const tarray& output)
                {
                        return -target.array() * (1 - output * target);
                }
        };

        using square_loss_t = regression_t<square_regression_t>;
        using ssquare_loss_t = sclassification_t<square_regression_t>;
        using msquare_loss_t = mclassification_t<square_regression_t>;
}
