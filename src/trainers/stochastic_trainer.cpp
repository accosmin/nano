#include "stochastic_trainer.h"
#include "core/timer.h"
#include "core/text.h"
#include "core/math/clamp.hpp"
#include "core/logger.h"
#include "core/thread.h"

namespace ncv
{
        /////////////////////////////////////////////////////////////////////////////////////////

        stochastic_trainer_t::stochastic_trainer_t(const string_t& params)
                :       m_iterations(text::from_params<size_t>(params, "iters", 1024)),
                        m_depth(text::from_params<size_t>(params, "depth", 4))
        {
                m_iterations = math::clamp(m_iterations, 100, 10000);
                m_depth = math::clamp(m_depth, 2, 16);
        }

        /////////////////////////////////////////////////////////////////////////////////////////

        scalar_t stochastic_trainer_t::sgd(
                const task_t& task, const samples_t& samples, const loss_t& loss, model_t& model, vector_t& x,
                scalar_t gamma, scalar_t lambda, size_t iterations, size_t evalsize) const
        {
                // SGD steps
                random_t<size_t> die(0, samples.size() - 1);
                for (size_t iteration = 0; iteration < iterations; iteration ++)
                {
                        const sample_t& sample = samples[die()];

                        model.zero_grad();
                        model.load_params(x);

                        const scalar_t f = ncv::lvgrad(task, sample, loss, model);
                        const scalar_t d = learning_rate(gamma, lambda, iteration);
                        const vector_t g = model.grad();

                        if (    std::isinf(f) || std::isinf(g.minCoeff()) || std::isinf(g.maxCoeff()) ||
                                std::isnan(f) || std::isnan(g.minCoeff()) || std::isnan(g.maxCoeff()))
                        {
                                return std::numeric_limits<scalar_t>::max();
                        }

                        x -= d * g;
                }

                // evaluate model
                samples_t esamples;
                for (size_t i = 0; i < evalsize; i ++)
                {
                        const sample_t& sample = samples[die()];

                        esamples.emplace_back(sample);
                }

                return ncv::lvalue_st(task, esamples, loss, model);
        }

        /////////////////////////////////////////////////////////////////////////////////////////

        bool stochastic_trainer_t::train(const task_t& task, const fold_t& fold, const loss_t& loss, model_t& model) const
        {
                if (fold.second != protocol::train)
                {
                        log_error() << "stochastic trainer: cannot only train models with training samples!";
                        return false;
                }

                // initialize the model
                model.resize(task);
                model.random_params();

                // prune training data
                const samples_t samples = ncv::prune_annotated(task, task.samples(fold));
                if (samples.empty())
                {
                        log_error() << "stochastic trainer: no annotated training samples!";
                        return false;
                }

                vector_t x(model.n_parameters());
                model.save_params(x);

                const size_t max_depth = 8;
                const size_t n_workers = std::max(size_t(4), 2 * ncv::n_threads());

                std::vector<state_t> states(n_workers);
                for (state_t& state : states)
                {
                        state.m_model = model.clone();
                }

                ncv::worker_pool_t worker_pool(1);

                // search the optimum learning parameter: each worker thread tests a value
                scalar_t min_log_lambda = 0.0;
                scalar_t max_log_lambda = 6.0;

                for (   size_t depth = 0, iterations = m_iterations, evalsize = iterations;
                        depth < m_depth;
                        depth ++, iterations = iterations * 2, evalsize = iterations)
                {
                        worker_pool_t::mutex_t mutex;

                        // create workers
                        const scalar_t delta_log_lambda = (max_log_lambda - min_log_lambda) / (n_workers - 1.0);
                        for (size_t n = 0; n < n_workers; n ++)
                        {
                                worker_pool.enqueue([=, &states, &task, &samples, &loss, &x, &mutex] ()
                                {
                                        state_t& state = states[n];
                                        state.m_log_lambda = min_log_lambda + delta_log_lambda * n;
                                        state.m_param = x;

                                        timer_t timer;
                                        state.m_lvalue = sgd(
                                                task, samples, loss, *state.m_model, state.m_param,
                                                1.0, make_lambda(state.m_log_lambda), iterations, evalsize);

                                        const worker_pool_t::lock_t lock(mutex);
                                        log_info() << "stochastic trainer: [depth = " << depth << "/" << max_depth
                                                   << ", lambda = " << make_lambda(state.m_log_lambda)
                                                   << ", param = [" << state.m_param.minCoeff() << ", "
                                                   << state.m_param.maxCoeff()
                                                   << "], loss = " << state.m_lvalue
                                                   << "] processed in " << timer.elapsed() << ".";
                                        timer.start();
                                });
                        }

                        worker_pool.wait();

                        // select the optimum learning parameter to expand
                        const state_t& opt_state = *std::min_element(states.begin(), states.end(),
                                [] (const state_t& state1, const state_t& state2)
                        {
                                return state1.m_lvalue < state2.m_lvalue;
                        });

                        const scalar_t opt_log_lambda = opt_state.m_log_lambda;
                        min_log_lambda = opt_log_lambda - delta_log_lambda;
                        max_log_lambda = opt_log_lambda + delta_log_lambda;

                        x = opt_state.m_param;

                        // log
                        log_info() << "stochastic trainer: optimum lambda = "
                                   << make_lambda(opt_state.m_log_lambda) << ".";
                }

                // update the model
                model.load_params(x);

                return true;
        }

        /////////////////////////////////////////////////////////////////////////////////////////
}
