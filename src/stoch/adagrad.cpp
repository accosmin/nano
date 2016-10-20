#include "loop.h"
#include "adagrad.h"

namespace nano
{
        stoch_adagrad_t::stoch_adagrad_t(const string_t& configuration) :
                stoch_optimizer_t(configuration)
        {
        }

        rstoch_optimizer_t stoch_adagrad_t::clone(const string_t& configuration) const
        {
                return std::make_unique<stoch_adagrad_t>(configuration);
        }

        rstoch_optimizer_t stoch_adagrad_t::clone() const
        {
                return std::make_unique<stoch_adagrad_t>(*this);
        }

        state_t stoch_adagrad_t::minimize(const stoch_params_t& param, const problem_t& problem, const vector_t& x0) const
        {
                return stoch_tune(this, param, problem, x0, make_alpha0s(), make_epsilons());
        }

        state_t stoch_adagrad_t::minimize(const stoch_params_t& param, const problem_t& problem, const vector_t& x0,
                const scalar_t alpha0, const scalar_t epsilon) const
        {
                assert(problem.size() == x0.size());

                // second-order gradient momentum
                vector_t gsum2 = vector_t::Zero(x0.size());

                const auto op_iter = [&] (state_t& cstate)
                {
                        // learning rate
                        const scalar_t alpha = alpha0;

                        // descent direction
                        gsum2.array() += cstate.g.array().square();

                        cstate.d = -cstate.g.array() / (epsilon + gsum2.array()).sqrt();

                        // update solution
                        cstate.update(problem, alpha);
                };

                // OK, assembly the optimizer
                return  stoch_loop(problem, param, state_t(problem, x0), op_iter,
                        {{"alpha0", alpha0}, {"epsilon", epsilon}});
        }
}

