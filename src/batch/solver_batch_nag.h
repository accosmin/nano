#pragma once

#include "solver_batch.h"

namespace nano
{
        ///
        /// \brief restart methods for Nesterov's accelerated gradient
        ///     see "Adaptive Restart for Accelerated Gradient Schemes",
        ///     by Brendan O’Donoghue & Emmanuel Candes, 2013
        ///
        enum class nag_restart
        {
                none,
                function,
                gradient
        };

        ///
        /// \brief Nesterov's accelerated gradient descent (with variations)
        ///     see "Adaptive Restart for Accelerated Gradient Schemes",
        ///     by Brendan O’Donoghue & Emmanuel Candes, 2013
        ///     see "Introductory lectures on convex optimization: A basic course",
        ///     by Y. Nesterov, 2004
        ///
        template <nag_restart trestart>
        class batch_nag_base_t final : public batch_solver_t
        {
        public:

                void to_json(json_t&) const final;
                void from_json(const json_t&) final;

                solver_state_t minimize(const batch_params_t&, const function_t&, const vector_t& x0) const final;

        private:

                // attributes
                scalar_t        m_alpha0{static_cast<scalar_t>(1e-2)};
                scalar_t        m_q{static_cast<scalar_t>(0.0)};
        };

        using batch_nag_t = batch_nag_base_t<nag_restart::none>;
        using batch_nagfr_t = batch_nag_base_t<nag_restart::function>;
        using batch_naggr_t = batch_nag_base_t<nag_restart::gradient>;
}
