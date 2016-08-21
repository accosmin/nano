#pragma once

#include "batch_optimizer.h"

namespace nano
{
        ///
        /// \brief gradient descent
        ///
        struct batch_gd_t : public batch_optimizer_t
        {
                NANO_MAKE_CLONABLE(batch_gd_t,
                        "gradient descent",
                        "ls_init=init-quadratic,ls_strat=backtrack-strong-wolfe,c1=1e-4,c2=0.1")

                ///
                /// \brief constructor
                ///
                batch_gd_t(const string_t& configuration = string_t());

                ///
                /// \brief minimize starting from the initial guess x0.
                ///
                virtual state_t minimize(const batch_params_t&, const problem_t&, const vector_t& x0) const override;
        };
}

