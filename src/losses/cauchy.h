#pragma once

#include "regression.h"
#include "classification.h"

namespace nano
{
        ///
        /// \brief robust-to-noise Cauchy loss: 1/2 * log(1 + x^2).
        ///
        /// usage:
        ///     - regression:           l(outputs, targets) = l(outputs - targets)
        ///     - classification:       l(outputs, targets) = l(1 - outputs * targets)
        ///
        struct cauchy_regression_t
        {
                static auto value(const vector_cmap_t& targets, const vector_cmap_t& outputs)
                {
                        return scalar_t(0.5) * ((targets - outputs).array().square() + 1).log().sum();
                }

                static auto vgrad(const vector_cmap_t& targets, const vector_cmap_t& outputs)
                {
                        return (outputs - targets).array() / (1 + (outputs - targets).array().square());
                }
        };

        struct cauchy_classification_t
        {
                static auto value(const vector_cmap_t& targets, const vector_cmap_t& outputs)
                {
                        return scalar_t(0.5) * ((targets - outputs).array().square() + 1).log().sum();
                }

                static auto vgrad(const vector_cmap_t& targets, const vector_cmap_t& outputs)
                {
                        return -targets.array() * (1 - outputs.array() * targets.array()) / (1 + (1 - outputs.array() * targets.array()).square());
                }
        };

        using cauchy_loss_t = regression_t<cauchy_regression_t>;
        using scauchy_loss_t = sclassification_t<cauchy_classification_t>;
        using mcauchy_loss_t = mclassification_t<cauchy_classification_t>;
}
