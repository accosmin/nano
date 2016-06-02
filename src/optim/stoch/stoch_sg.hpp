#pragma once

#include "lrate.hpp"
#include "stoch_loop.hpp"

namespace nano
{
        ///
        /// \brief stochastic gradient (descent)
        ///     see "Minimizing Finite Sums with the Stochastic Average Gradient",
        ///     by Mark Schmidth, Nicolas Le Roux, Francis Bach
        ///
        template
        <
                typename tproblem               ///< optimization problem
        >
        struct stoch_sg_t
        {
                using param_t = stoch_params_t<tproblem>;
                using tstate = typename param_t::tstate;
                using tscalar = typename param_t::tscalar;
                using tvector = typename param_t::tvector;

                ///
                /// \brief minimize starting from the initial guess x0
                ///
                tstate operator()(const param_t& param, const tproblem& problem, const tvector& x0) const
                {
                        const auto op = [&] (const auto... params)
                        {
                                return this->operator()(param.tunable(), problem, x0, params...);
                        };

                        const auto param0 = make_alpha0s<tscalar>();
                        const auto param1 = make_decays<tscalar>();
                        const auto config = nano::tune(op, param0, param1);
                        return operator()(param, problem, x0, config.param0(), config.param1());
                }

                ///
                /// \brief minimize starting from the initial guess x0
                ///
                tstate operator()(const param_t& param, const tproblem& problem, const tvector& x0,
                        const tscalar alpha0, const tscalar decay) const
                {
                        assert(problem.size() == x0.size());

                        // learning rate schedule
                        lrate_t<tscalar> lrate(alpha0, decay);

                        const auto op_iter = [&] (tstate& cstate)
                        {
                                // learning rate
                                const tscalar alpha = lrate.get();

                                // descent direction
                                cstate.d = -cstate.g;

                                // update solution
                                cstate.update(problem, alpha);
                        };

                        // OK, assembly the optimizer
                        return  stoch_loop(problem, param, tstate(problem, x0), op_iter,
                                {{"alpha0", alpha0}, {"decay", decay}});
                }
        };
}

