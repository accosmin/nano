#pragma once

#include "regression.h"
#include "classification.h"

namespace nano
{
        ///
        /// \brief robust-to-noise Cauchy loss: 1/2 * log(1 + x^2).
        ///
        /// usage:
        ///     - regression:           l(output, target) = l(output - target)
        ///     - classification:       l(output, target) = l(1 - output * target)
        ///
        struct cauchy_regression_t
        {
                template <typename tarray>
                static auto value(const tarray& target, const tarray& output)
                {
                        return scalar_t(0.5) * ((target - output).square() + 1).log().sum();
                }

                template <typename tarray>
                static auto vgrad(const tarray& target, const tarray& output)
                {
                        return (output - target) / (1 + (output - target).square());
                }
        };

        struct cauchy_classification_t
        {
                template <typename tarray>
                static auto value(const tarray& target, const tarray& output)
                {
                        return scalar_t(0.5) * ((target - output).square() + 1).log().sum();
                }

                template <typename tarray>
                static auto vgrad(const tarray& target, const tarray& output)
                {
                        return -target * (1 - output * target) / (1 + (1 - output * target).square());
                }
        };

        using cauchy_loss_t = regression_t<cauchy_regression_t>;
        using scauchy_loss_t = sclassification_t<cauchy_classification_t>;
        using mcauchy_loss_t = mclassification_t<cauchy_classification_t>;
}
