#include "stochastic_trainer.h"
#include "core/timer.h"
#include "core/text.h"
#include "core/math/clamp.hpp"
#include "core/thread/thread.h"
#include "core/logger.h"

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
                vector_t avg_x = x;

                // (A=average)SGD steps
                for (size_t iteration = 0; iteration < iterations; iteration ++)
                {                        
                        random_t<size_t> die(0, samples.size() - 1);
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
                        avg_x += x;
                }

                model.load_params(avg_x / (1.0 + iterations));

                // evaluate model
                random_t<size_t> die(0, samples.size() - 1);

                samples_t esamples(evalsize);
                for (sample_t& sample : esamples)
                {
                        sample = samples[die()];
                }

                return ncv::lvalue_st(task, esamples, loss, model);
        }

        /////////////////////////////////////////////////////////////////////////////////////////

        bool stochastic_trainer_t::train(
                const task_t& task, const fold_t& fold, const loss_t& loss, size_t nthreads, model_t& model) const
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

                const size_t n_workers = std::max(size_t(4), 2 * nthreads);
                ncv::thread_pool_t worker_pool(nthreads);

                // create worker buffers: (gamma, lambda) + loss values
                tensor::matrix_types_t<state_t>::matrix_t states(n_workers, n_workers);
                matrix_t lvalues(n_workers, n_workers);

                for (size_t i = 0; i < static_cast<size_t>(states.size()); i ++)
                {
                        states(i).m_model = model.clone();      // create model copies once!
                }

                // search the optimum learning parameters: each worker thread tests a value
                scalar_t min_log_lambda = -4.0;
                scalar_t max_log_lambda = +1.0;
                scalar_t min_log_gamma = -4.0;
                scalar_t max_log_gamma = +1.0;

                for (   size_t depth = 0, iterations = m_iterations, evalsize = iterations;
                        depth < m_depth;
                        depth ++, iterations = iterations, evalsize = 4 * m_iterations)
                {
                        thread_pool_t::mutex_t mutex;

                        // create workers (one for each learning parameter set)
                        const scalar_t delta_log_gamma = (max_log_gamma - min_log_gamma) / (n_workers - 1.0);
                        const scalar_t delta_log_lambda = (max_log_lambda - min_log_lambda) / (n_workers - 1.0);

                        for (size_t nl = 0; nl < n_workers; nl ++)              // lambda
                        {
                                for (size_t ng = 0; ng < n_workers; ng ++)      // gamma
                                {
                                        worker_pool.enqueue([=, &states, &lvalues, &task, &samples, &loss, &x, &mutex]()
                                        {
                                                // prepare parameters
                                                state_t& state = states(nl, ng);
                                                state.m_log_lambda = min_log_lambda + delta_log_lambda * nl;
                                                state.m_log_gamma = min_log_gamma + delta_log_gamma * ng;
                                                state.m_param = x;

                                                // average stochastic gradient descent (ASGD)
                                                timer_t timer;
                                                lvalues(nl, ng) = sgd(
                                                        task, samples, loss, *state.m_model, state.m_param,
                                                        make_param(state.m_log_gamma),
                                                        make_param(state.m_log_gamma),
                                                        iterations, evalsize);

                                                const thread_pool_t::lock_t lock(mutex);
                                                log_info() << "stochastic trainer: [depth = "
                                                           << (depth + 1) << "/" << m_depth
                                                           << ", param = (" << make_param(state.m_log_gamma)
                                                           << ", " << make_param(state.m_log_lambda)
                                                           << "), loss = " << lvalues(nl, ng)
                                                           << "] processed in " << timer.elapsed() << ".";
                                                timer.start();
                                        });
                                }
                        }

                        worker_pool.wait();

                        // select the optimum learning parameter to expand
                        matrix_t::Index nl, ng;
                        const scalar_t opt_lvalue = lvalues.minCoeff(&nl, &ng);
                        const state_t& opt_state = states(nl, ng);

                        update_range(min_log_gamma, max_log_gamma, opt_state.m_log_gamma, delta_log_gamma);
                        update_range(min_log_lambda, max_log_lambda, opt_state.m_log_lambda, delta_log_lambda);
                        x = opt_state.m_param;

                        // log
                        log_info() << "stochastic trainer: optimum param = (" << make_param(opt_state.m_log_gamma)
                                   << ", " << make_param(opt_state.m_log_lambda)
                                   << "), loss = " << opt_lvalue << ".";
                }

                // update the model
                model.load_params(x);

                return true;
        }

        /////////////////////////////////////////////////////////////////////////////////////////
}
