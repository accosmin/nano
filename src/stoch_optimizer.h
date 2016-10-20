#pragma once

#include "problem.h"
#include "manager.h"
#include "stoch/params.h"

namespace nano
{
        ///
        /// \brief stores registered prototypes
        ///
        class stoch_optimizer_t;
        using stoch_optimizer_manager_t = manager_t<stoch_optimizer_t>;
        using rstoch_optimizer_t = stoch_optimizer_manager_t::trobject;

        NANO_PUBLIC stoch_optimizer_manager_t& get_stoch_optimizers();

        ///
        /// \brief generic stochastic optimizer
        ///
        class NANO_PUBLIC stoch_optimizer_t : public clonable_t<stoch_optimizer_t>
        {
        public:

                ///
                /// \brief constructor
                ///
                explicit stoch_optimizer_t(const string_t& configuration = string_t()) :
                        clonable_t<stoch_optimizer_t>(configuration)
                {
                }

                ///
                /// \brief minimize starting from the initial point x0.
                ///
                virtual state_t minimize(const stoch_params_t&, const problem_t&, const vector_t& x0) const = 0;
        };
}
