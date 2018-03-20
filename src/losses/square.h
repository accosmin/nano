#pragma once

#include "regression.h"
#include "classification.h"

namespace nano
{
        ///
        /// \brief square loss: l(x) = 1/2 * x^x.
        ///
        /// usage:
        ///     - regression:           l(outputs, targets) = l(outputs - targets)
        ///     - classification:       l(outputs, targets) = l(1 - outputs * targets)
        ///
        struct square_regression_t
        {
                static auto value(const vector_cmap_t& targets, const vector_cmap_t& outputs)
                {
                        return scalar_t(0.5) * (outputs - targets).array().square().sum();
                }

                static auto vgrad(const vector_cmap_t& targets, const vector_cmap_t& outputs)
                {
                        return outputs - targets;
                }
        };

        struct square_classification_t
        {
                static auto value(const vector_cmap_t& targets, const vector_cmap_t& outputs)
                {
                        return scalar_t(0.5) * (1 - outputs.array() * targets.array()).square().sum();
                }

                static auto vgrad(const vector_cmap_t& targets, const vector_cmap_t& outputs)
                {
                        return -targets.array() * (1 - outputs.array() * targets.array());
                }
        };

        using square_loss_t = regression_t<square_regression_t>;
        using ssquare_loss_t = sclassification_t<square_regression_t>;
        using msquare_loss_t = mclassification_t<square_regression_t>;
}
