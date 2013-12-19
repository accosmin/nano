#ifndef NANOCV_OPTIMIZE_OPTIMIZER_GD_HPP
#define NANOCV_OPTIMIZE_OPTIMIZER_GD_HPP

#include <cassert>
#include <cmath>

namespace ncv
{
        namespace optimize
        {
                /////////////////////////////////////////////////////////////////////////////////////////
                // stochastic gradient descent starting from the initial value (guess) x0.
                /////////////////////////////////////////////////////////////////////////////////////////

                template
                <
                        typename tscalar,
                        typename tsize
                >
                tscalar sgd_learning_rate(tscalar gamma, tscalar lambda, tsize iteration)
                {
                        // learning rate recommended by Bottou
                        return gamma / (1 + gamma * lambda * iteration);
                }

                template
                <
                        typename tproblem,

                        // dependent types
                        typename tscalar = typename tproblem::tscalar,
                        typename tsize = typename tproblem::tsize,
                        typename tvector = typename tproblem::tvector,
                        typename tresult = typename tproblem::tresult,
                        typename tstate = typename tproblem::tstate
                >
                tresult sgd(
                        const tproblem& problem,
                        const tvector& x0,
                        tsize max_iterations,
                        tscalar gamma,
                        tscalar lambda)
                {
                        assert(problem.size() == static_cast<tsize>(x0.size()));

                        tresult result(problem.size());
                        tstate cstate(problem, x0);

//                        scalar_t stochastic_trainer_t::sgd(
//                                const task_t& task, const samples_t& samples, const loss_t& loss, model_t& model, vector_t& x,
//                                scalar_t gamma, scalar_t lambda, size_t iterations, size_t evalsize) const
//                        {
//                                vector_t avg_x = x;

//                                // (A=average)SGD steps
//                                size_t iteration = 0;
//                                ncv::uniform_sample(samples, iterations, [&] (const sample_t& sample)
//                                {
//                                        model.load_params(x);

//                                        const scalar_t f = ncv::lvgrad(task, sample, loss, model);
//                                        const scalar_t d = learning_rate(gamma, lambda, iteration);
//                                        const vector_t g = model.grad();

//                                        if (    !std::isinf(f) && !std::isinf(g.minCoeff()) && !std::isinf(g.maxCoeff()) &&
//                                                !std::isnan(f) && !std::isnan(g.minCoeff()) && !std::isnan(g.maxCoeff()))
//                                        {
//                                                x.noalias() -= d * g;
//                                        }

//                                        avg_x.noalias() += x;
//                                        ++ iteration;
//                                });

//                                x = avg_x / (1.0 + iterations);
//                                model.load_params(x);

//                                // evaluate model
//                                const samples_t esamples = (evalsize == samples.size()) ?
//                                                samples : ncv::uniform_sample(samples, evalsize);
//                                return ncv::lvalue_st(task, esamples, loss, model);
//                        }

//                        // iterate until convergence
//                        for (tsize i = 0; i < max_iterations; i ++)
//                        {
//                                result.update(problem, cstate);

//                                // check convergence
//                                if (cstate.converged(epsilon))
//                                {
//                                        break;
//                                }

//                                // descent direction
//                                cstate.d = -cstate.g;

//                                // update solution
//                                const tscalar t = ls_armijo(problem, cstate, op_wlog);
//                                if (t < std::numeric_limits<tscalar>::epsilon())
//                                {
//                                        if (op_elog)
//                                        {
//                                                op_elog("line-search failed for GD!");
//                                        }
//                                        break;
//                                }
//                                cstate.update(problem, t);
//                        }

                        return result;
                }
        }
}

#endif // NANOCV_OPTIMIZE_OPTIMIZER_GD_HPP
