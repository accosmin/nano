#pragma once

#include "batch_optimizer.h"

namespace nano
{
        ///
        /// \brief limited memory bfgs (l-bfgs)
        ///
        struct batch_lbfgs_t : public batch_optimizer_t
        {
                NANO_MAKE_CLONABLE(batch_lbfgs_t)

                ///
                /// \brief constructor
                ///
                batch_lbfgs_t(const string_t& configuration = string_t());

                ///
                /// \brief minimize starting from the initial guess x0.
                ///
                virtual state_t minimize(const batch_params_t&, const problem_t&, const vector_t& x0) const override final;
        };
}

