#include "stochastic_trainer.h"
#include "util/timer.h"
#include "util/logger.h"
#include "util/math.hpp"
#include "util/usampler.hpp"
#include "optimize/opt_sgd.hpp"
#include "optimize/opt_asgd.hpp"
#include "text.h"
#include "thread/thread_pool.h"

namespace ncv
{
        /////////////////////////////////////////////////////////////////////////////////////////

        stochastic_trainer_t::stochastic_trainer_t(const string_t& params)
                :       m_optimizer(text::from_params<string_t>(params, "opt", "asgd")),
                        m_epoch(text::from_params<size_t>(params, "epoch", 4))
        {
                m_epoch = math::clamp(m_epoch, 2, 16);
        }

        /////////////////////////////////////////////////////////////////////////////////////////

        scalar_t stochastic_trainer_t::sgd(
                const task_t& task, const samples_t& samples, const loss_t& loss, model_t& model, vector_t& x,
                scalar_t gamma, scalar_t lambda, size_t iterations, size_t evalsize) const
        {
                random_t<size_t> rng(0, samples.size());

                // optimization problem: size
                auto fn_size = [&] ()
                {
                        return model.n_parameters();
                };

                // optimization problem: function value
                auto fn_fval = [&] (const vector_t& x)
                {
                        model.load_params(x);

                        const bool all = evalsize == samples.size();
                        const samples_t usamples = all ? samples : ncv::uniform_sample(samples, evalsize);
                        return ncv::lvalue_st(task, usamples, loss, model);
                };

                // optimization problem: function value & gradient
                auto fn_fval_grad = [&] (const vector_t& x, vector_t& gx)
                {
                        model.load_params(x);

                        const sample_t& sample = samples[rng() % samples.size()];
                        return ncv::lvgrad(task, sample, loss, model);
                };

                // assembly optimization problem & optimize the model
                const opt_problem_t problem(fn_size, fn_fval, fn_fval_grad);
                opt_result_t res;

                if (text::iequals(m_optimizer, "asgd"))
                {
                        res = optimize::asgd(problem, x, iterations, gamma, lambda);
                }
                else if (text::iequals(m_optimizer, "sgd"))
                {
                        res = optimize::sgd(problem, x, iterations, gamma, lambda);
                }
                else
                {
                        log_error() << "stochastic trainer: invalid optimization method <" << m_optimizer << ">!";
                }

                x = res.optimum().x;
                return res.optimum().f;
        }

        /////////////////////////////////////////////////////////////////////////////////////////

        struct state_t
        {
                void init(const model_t& model, const vector_t& x, scalar_t log_gamma, scalar_t log_lambda)
                {
                        m_model = model.clone();
                        m_param = x;
                        m_log_gamma = log_gamma;
                        m_log_lambda = log_lambda;
                }

                scalar_t gamma() const { return std::pow(10.0, m_log_gamma); }
                scalar_t lambda() const { return std::pow(10.0, m_log_lambda); }

                scalar_t        m_log_lambda;
                scalar_t        m_log_gamma;
                rmodel_t        m_model;
                vector_t        m_param;
        };

        /////////////////////////////////////////////////////////////////////////////////////////

        void stochastic_trainer_t::tune_sgd(
                const task_t& task, const samples_t& samples, const loss_t& loss,
                size_t nthreads, model_t& model, vector_t& x,
                scalar_t& opt_log_gamma, scalar_t& opt_log_lambda) const
        {
                ncv::thread_pool_t worker_pool(nthreads);
                thread_pool_t::mutex_t mutex;

                const size_t n_workers = std::max(size_t(4), worker_pool.n_workers());

                // create worker buffers: (gamma, lambda) + loss values
                tensor::matrix_types_t<state_t>::tmatrix states(n_workers, n_workers);
                tensor::matrix_types_t<scalar_t>::tmatrix lvalues(n_workers, n_workers);

                // search the optimum learning parameters: each worker thread tests a value
                scalar_t min_log_gamma = -4.0;
                scalar_t max_log_gamma = +1.0;
                scalar_t min_log_lambda = -4.0;
                scalar_t max_log_lambda = +1.0;

                const scalar_t min_delta_log = 1e-1;
                const size_t tune_iters = 1024;
                const size_t tune_evals = 4096;

                while ( max_log_gamma - min_log_gamma > min_delta_log &&
                        max_log_lambda - min_log_lambda > min_delta_log)
                {
                        // create workers (one for each learning parameter set)
                        const scalar_t delta_log_gamma = (max_log_gamma - min_log_gamma) / (n_workers - 1.0);
                        const scalar_t delta_log_lambda = (max_log_lambda - min_log_lambda) / (n_workers - 1.0);

                        for (size_t nl = 0; nl < n_workers; nl ++)              // lambda
                        {
                                for (size_t ng = 0; ng < n_workers; ng ++)      // gamma
                                {
                                        worker_pool.enqueue(
                                                [=, &states, &lvalues, &task, &samples, &loss, &model, &x, &mutex]()
                                        {
                                                // prepare parameters
                                                state_t& state = states(nl, ng);
                                                state.init(model, x,
                                                           min_log_gamma + delta_log_gamma * ng,
                                                           min_log_lambda + delta_log_lambda * nl);

                                                // average stochastic gradient descent (ASGD)
                                                timer_t timer;
                                                const scalar_t lvalue = lvalues(nl, ng) = sgd(
                                                        task, samples, loss, *state.m_model, state.m_param,
                                                        state.gamma(), state.lambda(), tune_iters, tune_evals);

                                                const thread_pool_t::lock_t lock(mutex);
                                                log_info() << "stochastic trainer: tuning step [param = ("
                                                           << state.gamma() << ", " << state.lambda()
                                                           << "), loss = " << lvalue
                                                           << "] processed in " << timer.elapsed() << ".";
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
                        log_info() << "stochastic trainer: tuning step finalized [param = ("
                                   << opt_state.gamma() << ", " << opt_state.lambda()
                                   << "), loss = " << opt_lvalue << "].";
                }

                // OK
                opt_log_gamma = 0.5 * (min_log_gamma + max_log_gamma);
                opt_log_lambda = 0.5 * (min_log_lambda + max_log_lambda);
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

                // tune SGD parameters
                scalar_t opt_log_gamma, opt_log_lambda;
                tune_sgd(task, samples, loss, nthreads, model, x, opt_log_gamma, opt_log_lambda);

                // optimize using the optimal SGD parameters
                const size_t opt_iters = m_epoch * samples.size();
                const size_t opt_evals = samples.size();

                ncv::thread_pool_t worker_pool(nthreads);
                thread_pool_t::mutex_t mutex;

                const size_t n_workers = worker_pool.n_workers();
                std::vector<state_t> states(n_workers);
                std::vector<scalar_t> lvalues(n_workers);

                for (size_t n = 0; n < n_workers; n ++)
                {
                        worker_pool.enqueue([=, &states, &lvalues, &task, &samples, &loss, &model, &x, &mutex]()
                        {
                                // prepare parameters
                                state_t& state = states[n];
                                state.init(model, x, opt_log_gamma, opt_log_lambda);

                                // average stochastic gradient descent (ASGD)
                                timer_t timer;
                                const scalar_t lvalue = lvalues[n] = sgd(
                                        task, samples, loss, *state.m_model, state.m_param,
                                        state.gamma(), state.lambda(), opt_iters, opt_evals);

                                const thread_pool_t::lock_t lock(mutex);
                                log_info() << "stochastic trainer: optimization step [param = ("
                                           << state.gamma() << ", " << state.lambda() << "), loss = " << lvalue
                                           << "] processed in " << timer.elapsed() << ".";
                        });
                }

                worker_pool.wait();

                // update the model
                const size_t opt_index = std::min_element(lvalues.begin(), lvalues.end()) - lvalues.begin();
                model.load_params(states[opt_index].m_param);

                return true;
        }

        /////////////////////////////////////////////////////////////////////////////////////////
}
