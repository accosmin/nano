#pragma once

#include "stoch_optimizer.h"

namespace nano
{
        ///
        /// \brief restart methods for Nesterov's accelerated gradient
        ///     see "Adaptive Restart for Accelerated Gradient Schemes",
        ///     by Brendan O’Donoghue & Emmanuel Candes, 2013
        ///
        enum class ag_restart
        {
                none,
                function,
                gradient
        };

        ///
        /// \brief stochastic Nesterov's accelerated gradient (descent)
        ///     see "Adaptive Restart for Accelerated Gradient Schemes",
        ///     by Brendan O’Donoghue & Emmanuel Candes, 2013
        ///
        template <ag_restart trestart>
        struct stoch_ag_base_t final : public stoch_optimizer_t
        {
                explicit stoch_ag_base_t(const string_t& configuration = string_t());

                virtual rstoch_optimizer_t clone() const override;

                virtual state_t minimize(const stoch_params_t&, const problem_t&, const vector_t& x0) const override;

                ///
                /// \brief minimize starting from the initial guess x0 using the given hyper-parameters.
                ///
                state_t minimize(const stoch_params_t& param, const problem_t& problem, const vector_t& x0,
                        const scalar_t alpha0, const scalar_t q) const;
        };

        using stoch_ag_t = stoch_ag_base_t<ag_restart::none>;
        using stoch_agfr_t = stoch_ag_base_t<ag_restart::function>;
        using stoch_aggr_t = stoch_ag_base_t<ag_restart::gradient>;
}

