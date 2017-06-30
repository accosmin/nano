#pragma once

#include "factory.h"
#include "function.h"
#include "stoch/params.h"

namespace nano
{
        ///
        /// \brief stores registered prototypes
        ///
        struct stoch_solver_t;
        using stoch_solver_factory_t = factory_t<stoch_solver_t>;
        using rstoch_solver_t = stoch_solver_factory_t::trobject;

        NANO_PUBLIC stoch_solver_factory_t& get_stoch_solvers();

        ///
        /// \brief generic stochastic optimizer.
        ///
        struct NANO_PUBLIC stoch_solver_t : public configurable_t
        {
                using configurable_t::configurable_t;

                ///
                /// \brief minimize starting from the initial point x0.
                /// NB: all its hyper-parameters are tuned.
                ///
                virtual function_state_t minimize(const stoch_params_t&, const function_t&, const vector_t& x0) const = 0;
        };
}
