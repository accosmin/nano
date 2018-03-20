#pragma once

#include "classification.h"

namespace nano
{
        ///
        /// \brief multi-class exponential loss: sum(exp(-targets_k * outputs_k), k).
        ///
        struct exponential_t
        {
                static auto value(const vector_cmap_t& targets, const vector_cmap_t& outputs)
                {
                        return (-targets.array() * outputs.array()).exp().sum();
                }

                static auto vgrad(const vector_cmap_t& targets, const vector_cmap_t& outputs)
                {
                        return -targets.array() * (-targets.array() * outputs.array()).exp();
                }
        };

        using mexponential_loss_t = mclassification_t<exponential_t>;
        using sexponential_loss_t = sclassification_t<exponential_t>;
}
