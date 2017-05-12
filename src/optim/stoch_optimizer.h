#pragma once

#include "manager.h"
#include "function.h"
#include "stoch/params.h"

namespace nano
{
        ///
        /// \brief stores registered prototypes
        ///
        struct stoch_optimizer_t;
        using stoch_optimizer_manager_t = manager_t<stoch_optimizer_t>;
        using rstoch_optimizer_t = stoch_optimizer_manager_t::trobject;

        NANO_PUBLIC stoch_optimizer_manager_t& get_stoch_optimizers();

        ///
        /// \brief generic stochastic optimizer
        ///
        struct NANO_PUBLIC stoch_optimizer_t : public configurable_t
        {
                using configurable_t::configurable_t;

                ///
                /// \brief minimize starting from the initial point x0
                ///
                virtual state_t minimize(const stoch_params_t&, const function_t&, const vector_t& x0) const = 0;
        };
}
