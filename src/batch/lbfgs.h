#pragma once

#include "batch_optimizer.h"

namespace nano
{
        ///
        /// \brief limited memory bfgs (l-bfgs)
        ///
        struct batch_lbfgs_t final : public batch_optimizer_t
        {
                batch_lbfgs_t(const string_t& configuration = string_t());

                virtual rbatch_optimizer_t clone() const override;

                virtual state_t minimize(const batch_params_t&, const problem_t&, const vector_t& x0) const override;
        };
}

