#include "stochastic_trainer.h"
#include "util/timer.h"
#include "util/logger.h"
#include "util/math.hpp"
#include "util/usampler.hpp"
#include "optimize/opt_sgd.hpp"
#include "optimize/opt_asgd.hpp"
#include "text.h"
#include "trainer_data.h"

namespace ncv
{
        /////////////////////////////////////////////////////////////////////////////////////////

        struct stochastic_state_t
        {
                stochastic_state_t() :     m_log_lambda(-4.0),
                                m_log_gamma(-4.0),
                                m_value(std::numeric_limits<scalar_t>::max()),
                                m_error(std::numeric_limits<scalar_t>::max())
                {
                }

                void init(const model_t& model)
                {
                        m_ldata.init(model);
                        m_gdata.init(model);
                }

                void init(const vector_t& x, scalar_t log_gamma, scalar_t log_lambda)
                {
                        m_param = x;
                        m_log_gamma = log_gamma;
                        m_log_lambda = log_lambda;
                }

                scalar_t gamma() const { return std::pow(10.0, m_log_gamma); }
                scalar_t lambda() const { return std::pow(10.0, m_log_lambda); }

                scalar_t                m_log_lambda;
                scalar_t                m_log_gamma;
                vector_t                m_param;
                scalar_t                m_value;
                scalar_t                m_error;

                trainer_data_skipgrad_t m_ldata;
                trainer_data_withgrad_t m_gdata;
        };

        /////////////////////////////////////////////////////////////////////////////////////////

        bool operator<(const stochastic_state_t& state1, const stochastic_state_t& state2)
        {
                return state1.m_error < state2.m_error;
        }

        /////////////////////////////////////////////////////////////////////////////////////////

        stochastic_trainer_t::stochastic_trainer_t(const string_t& params)
                :       m_optimizer(text::from_params<string_t>(params, "opt", "asgd")),
                        m_epoch(text::from_params<size_t>(params, "epoch", 4))
        {
                m_epoch = math::clamp(m_epoch, 1, 16);
        }

        /////////////////////////////////////////////////////////////////////////////////////////

        void stochastic_trainer_t::sgd(
                const task_t& task, const samples_t& samples, const loss_t& loss,
                size_t iterations, size_t evalsize, stochastic_state_t& state) const
        {
                random_t<size_t> rng(0, samples.size());

                // optimization problem: size
                auto fn_size = [&] ()
                {
                        return state.m_ldata.n_parameters();
                };

                // optimization problem: function value
                auto fn_fval = [&] (const vector_t& x)
                {
                        const samples_t usamples = evalsize == samples.size() ?
                                samples : ncv::uniform_sample(samples, evalsize, rng);

                        state.m_ldata.load_params(x);
                        state.m_ldata.update_st(task, usamples, loss);

                        state.m_param = x;
                        state.m_value = state.m_ldata.value();
                        state.m_error = state.m_ldata.error();
                        return state.m_value;
                };

                // optimization problem: function value & gradient
                auto fn_fval_grad = [&] (const vector_t& x, vector_t& gx)
                {
                        state.m_gdata.load_params(x);
                        state.m_gdata.update(task, samples[rng() % samples.size()], loss);

                        gx = state.m_gdata.vgrad();
                        return state.m_gdata.value();
                };

                // assembly optimization problem & optimize the model
                const opt_problem_t problem(fn_size, fn_fval, fn_fval_grad);

                if (text::iequals(m_optimizer, "asgd"))
                {
                        optimize::asgd(problem, state.m_param, iterations, state.gamma(), state.lambda());
                }
                else if (text::iequals(m_optimizer, "sgd"))
                {
                        optimize::sgd(problem, state.m_param, iterations, state.gamma(), state.lambda());
                }
                else
                {
                        log_error() << "stochastic trainer: invalid optimization method <" << m_optimizer << ">!";
                }
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

                vector_t x = model.params();

                // create worker buffers: optimize given a (gamma, lambda) parametrization
                ncv::thread_pool_t worker_pool(nthreads);
                thread_pool_t::mutex_t mutex;

                const size_t n_workers = std::max(size_t(4), worker_pool.n_workers());

                std::vector<stochastic_state_t> states(n_workers * n_workers);
                for (size_t i = 0; i < states.size(); i ++)
                {
                        states[i].init(model);
                }

                // search the optimum learning parameters: each worker thread tests a value
                scalar_t min_log_gamma = -4.0;
                scalar_t max_log_gamma = +1.0;
                scalar_t min_log_lambda = -4.0;
                scalar_t max_log_lambda = +1.0;
                scalar_t min_delta = +0.02;

                const size_t tune_iters = std::max(size_t(1024), samples.size() / 64);
                const size_t tune_evals = 4 * tune_iters;
                const size_t opt_iters = m_epoch * samples.size();
                const size_t opt_evals = samples.size();

                for (size_t e = 0, tuned = 0; !tuned; e ++)
                {
                        tuned = max_log_gamma < min_log_gamma + min_delta &&
                                max_log_lambda < min_log_lambda + min_delta;

                        const size_t iters = tuned ? opt_iters : tune_iters;
                        const size_t evals = tuned ? opt_evals : tune_evals;

                        // create workers (one for each learning parameter set)
                        const scalar_t delta_log_gamma = (max_log_gamma - min_log_gamma) / (n_workers - 1.0);
                        const scalar_t delta_log_lambda = (max_log_lambda - min_log_lambda) / (n_workers - 1.0);

                        for (size_t nl = 0; nl < n_workers; nl ++)              // lambda
                        {
                                for (size_t ng = 0; ng < n_workers; ng ++)      // gamma
                                {
                                        worker_pool.enqueue([=, &states, &task, &samples, &loss, &model, &x, &mutex]()
                                        {
                                                // prepare parameters
                                                stochastic_state_t& state = states[nl * n_workers + ng];
                                                state.init(x,
                                                           min_log_gamma + delta_log_gamma * ng,
                                                           min_log_lambda + delta_log_lambda * nl);

                                                // stochastic optimization
                                                timer_t timer;
                                                sgd(task, samples, loss, iters, evals, state);

                                                const thread_pool_t::lock_t lock(mutex);
                                                log_info() << "stochastic trainer: step [" << (e + 1)
                                                           << ", param = (" << state.gamma() << ", " << state.lambda()
                                                           << "), loss = " << state.m_value << "/" << state.m_error
                                                           << "] done in " << timer.elapsed() << ".";
                                        });
                                }
                        }

                        worker_pool.wait();

                        // select the optimum learning parameter to expand
                        std::sort(states.begin(), states.end());
                        const stochastic_state_t& opt_state = states[0];

                        update_range(min_log_gamma, max_log_gamma, opt_state.m_log_gamma, delta_log_gamma);
                        update_range(min_log_lambda, max_log_lambda, opt_state.m_log_lambda, delta_log_lambda);
                        x = opt_state.m_param;

                        model.load_params(x);

                        // log
                        log_info() << "stochastic trainer: step finalized [" << (e + 1)
                                   << ", param = (" << opt_state.gamma() << ", " << opt_state.lambda()
                                   << "), loss* = " << opt_state.m_value << "/" << opt_state.m_error
                                   << ", iters = " << iters << "/" << evals << "].";
                }

                // OK
                return true;
        }

        /////////////////////////////////////////////////////////////////////////////////////////
}
