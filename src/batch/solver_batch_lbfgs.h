#pragma once

#include "solver_batch.h"

namespace nano
{
        ///
        /// \brief limited memory bfgs (l-bfgs)
        ///
        struct batch_lbfgs_t final : public batch_solver_t
        {
                explicit batch_lbfgs_t(const string_t& params = string_t());

                function_state_t minimize(const batch_params_t&, const function_t&, const vector_t& x0) const override;

                function_state_t minimize(const batch_params_t&, const function_t&, const vector_t& x0,
                        const ls_initializer, const ls_strategy, const scalar_t c1, const scalar_t c2) const;
        };
}
