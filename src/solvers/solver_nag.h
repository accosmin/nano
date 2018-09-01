#pragma once

#include "solver.h"
#include "lsearch.h"
#include "core/numeric.h"

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
        class solver_nag_base_t final : public solver_t
        {
        public:

                solver_nag_base_t() = default;

                tuner_t tuner() const final;
                void to_json(json_t&) const final;
                void from_json(const json_t&) final;

                solver_state_t minimize(
                        const size_t max_iterations, const scalar_t epsilon,
                        const function_t&, const vector_t& x0,
                        const logger_t& logger = logger_t()) const final;

        private:

                // attributes
                lsearch_t::initializer  m_init{lsearch_t::initializer::unit};
                lsearch_t::strategy     m_strat{lsearch_t::strategy::cg_descent};
                scalar_t                m_c1{epsilon2<scalar_t>()};
                scalar_t                m_c2{static_cast<scalar_t>(9e-1)};
                scalar_t                m_q{static_cast<scalar_t>(0.0)};
        };

        using solver_nag_t = solver_nag_base_t<nag_restart::none>;
        using solver_nagfr_t = solver_nag_base_t<nag_restart::function>;
        using solver_naggr_t = solver_nag_base_t<nag_restart::gradient>;
}
