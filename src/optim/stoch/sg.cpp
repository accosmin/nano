#include "sg.h"
#include "lrate.h"
#include "loop.hpp"

namespace nano
{
        state_t stoch_sg_t::operator()(const stoch_params_t& param, const problem_t& problem, const vector_t& x0) const
        {
                const auto op = [&] (const auto... hypers)
                {
                        return this->operator()(param, problem, x0, hypers...);
                };

                return nano::tune(op, make_alpha0s(), make_decays()).optimum();
        }

        state_t stoch_sg_t::operator()(const stoch_params_t& param, const problem_t& problem, const vector_t& x0,
                const scalar_t alpha0, const scalar_t decay) const
        {
                assert(problem.size() == x0.size());

                // learning rate schedule
                lrate_t lrate(alpha0, decay);

                const auto op_iter = [&] (state_t& cstate)
                {
                        // learning rate
                        const scalar_t alpha = lrate.get();

                        // descent direction
                        cstate.d = -cstate.g;

                        // update solution
                        cstate.update(problem, alpha);
                };

                // OK, assembly the optimizer
                return  stoch_loop(problem, param, state_t(problem, x0), op_iter,
                        {{"alpha0", alpha0}, {"decay", decay}});
        }
}

