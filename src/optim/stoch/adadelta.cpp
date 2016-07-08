#include "adadelta.h"
#include "loop.hpp"
#include "math/momentum.hpp"

namespace nano
{
        state_t stoch_adadelta_t::operator()(const stoch_params_t& param, const problem_t& problem, const vector_t& x0) const
        {
                const auto op = [&] (const auto... hypers)
                {
                        return this->operator()(param.tunable(), problem, x0, hypers...);
                };

                const auto config = nano::tune(op, make_momenta(), make_epsilons());
                return operator()(param.tuned(), problem, x0, config.param0(), config.param1());
        }

        state_t stoch_adadelta_t::operator()(const stoch_params_t& param, const problem_t& problem, const vector_t& x0,
                const scalar_t momentum, const scalar_t epsilon) const
        {
                assert(problem.size() == x0.size());

                // second-order momentum of the gradient
                momentum_vector_t<vector_t> gavg(momentum, x0.size());

                // second-order momentum of the step updates
                momentum_vector_t<vector_t> davg(momentum, x0.size());

                const auto op_iter = [&] (state_t& cstate)
                {
                        // learning rate
                        const scalar_t alpha = 1;

                        // descent direction
                        gavg.update(cstate.g.array().square());

                        cstate.d = -cstate.g.array() *
                                   (epsilon + davg.value().array()).sqrt() /
                                   (epsilon + gavg.value().array()).sqrt();

                        davg.update(cstate.d.array().square());

                        // update solution
                        cstate.update(problem, alpha);
                };

                // OK, assembly the optimizer
                return  stoch_loop(problem, param, state_t(problem, x0), op_iter,
                        {{"momentum", momentum}, {"epsilon", epsilon}});
        }
}

