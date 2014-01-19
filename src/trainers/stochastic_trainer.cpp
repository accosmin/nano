#include "stochastic_trainer.h"
#include "util/timer.h"
#include "util/logger.h"
#include "util/math.hpp"
#include "util/usampler.hpp"
#include "optimize/opt_sgd.hpp"
#include "optimize/opt_asgd.hpp"
#include "text.h"
#include "trainer_data.h"
#include "trainer_state.h"

namespace ncv
{
        /////////////////////////////////////////////////////////////////////////////////////////

//        todo: this should be merged into trainer_state_t!!!

        struct stochastic_state_t
        {
                stochastic_state_t() :
                        m_log_lambda(-4.0),
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
                        m_epochs(text::from_params<size_t>(params, "epoch", 16))
        {
                m_epochs = math::clamp(m_epochs, 1, 256);
        }

        /////////////////////////////////////////////////////////////////////////////////////////

        void stochastic_trainer_t::sgd(
                const task_t& task, const samples_t& tsamples, const samples_t& vsamples, const loss_t& loss,
                size_t iterations, size_t evalsize, stochastic_state_t& state) const
        {
//                todo: no need for lambdas here, move the (a)sgd code from optimize here,
//                                periodically check the training and the validation error

                random_t<size_t> trng(0, tsamples.size());
                random_t<size_t> vrng(0, vsamples.size());

                // optimization problem: size
                auto fn_size = [&] ()
                {
                        return state.m_ldata.n_parameters();
                };

                // optimization problem: function value (validation data)
                auto fn_fval = [&] (const vector_t& x)
                {
                        const samples_t usamples = evalsize == vsamples.size() ?
                                vsamples : ncv::uniform_sample(vsamples, evalsize, vrng);

                        state.m_ldata.load_params(x);
                        state.m_ldata.update_st(task, usamples, loss);

                        state.m_param = x;
                        state.m_value = state.m_ldata.value();
                        state.m_error = state.m_ldata.error();
                        return state.m_value;
                };

                // optimization problem: function value & gradient (training data)
                auto fn_fval_grad = [&] (const vector_t& x, vector_t& gx)
                {
                        state.m_gdata.load_params(x);
                        state.m_gdata.update(task, tsamples[trng() % tsamples.size()], loss);

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

                // prune training & validation data
                const samples_t samples = ncv::prune_annotated(task, task.samples(fold));
                if (samples.empty())
                {
                        log_error() << "stochastic trainer: no annotated training samples!";
                        return false;
                }

                samples_t tsamples, vsamples;
                ncv::uniform_split(samples, size_t(80), random_t<size_t>(0, samples.size()), tsamples, vsamples);

                const vector_t x0 = model.params();

                // create worker buffers: optimize given a (gamma, lambda) parametrization
                ncv::thread_pool_t wpool(nthreads);
                thread_pool_t::mutex_t mutex;

                const size_t n_workers = std::max(size_t(4), wpool.n_workers());

                std::vector<stochastic_state_t> states;

                // tune the learning parameters
                const scalar_t min_log_gamma = -4.0;
                const scalar_t max_log_gamma = -1.0;
                const scalar_t min_log_lambda = -4.0;
                const scalar_t max_log_lambda = -1.0;
                const scalar_t log_delta = +0.5;

                scalar_t opt_log_gamma = min_log_gamma;
                scalar_t opt_log_lambda = min_log_lambda;
                {
                        const size_t iters = std::max(size_t(1024), tsamples.size() / 64);
                        const size_t evals = 4 * iters;

                        // create workers (one for each learning parameter set)
                        for (scalar_t log_gamma = min_log_gamma; log_gamma < max_log_gamma; log_gamma += log_delta)
                        for (scalar_t log_lambda = min_log_lambda; log_lambda < max_log_lambda; log_lambda += log_delta)
                        {
                                wpool.enqueue([=, &states, &task, &tsamples, &vsamples, &loss, &model, &x0, &mutex]()
                                {
                                        // prepare parameters
                                        stochastic_state_t state;
                                        state.init(model);
                                        state.init(x0, log_gamma, log_lambda);

                                        // stochastic optimization
                                        timer_t timer;
                                        sgd(task, tsamples, vsamples, loss, iters, evals, state);

                                        const thread_pool_t::lock_t lock(mutex);
                                        states.push_back(state);

                                        log_info() << "stochastic trainer: tune ["
                                                   << "param = (" << state.gamma() << ", " << state.lambda()
                                                   << "), loss = " << state.m_value << "/" << state.m_error
                                                   << "] done in " << timer.elapsed() << ".";
                                });
                        }

                        wpool.wait();

                        // select the optimum learning parameter to expand
                        std::sort(states.begin(), states.end());
                        const stochastic_state_t& opt_state = states[0];

                        opt_log_gamma = opt_state.m_log_gamma;
                        opt_log_lambda = opt_state.m_log_lambda;

                        // log
                        log_info() << "stochastic trainer: tune ["
                                   << "param = (" << opt_state.gamma() << ", " << opt_state.lambda()
                                   << "), loss* = " << opt_state.m_value << "/" << opt_state.m_error
                                   << ", iters = " << iters << "/" << evals << "].";
                }

                // optimize the model (with the tuned parameters)
                {
                        const size_t iters = m_epochs * tsamples.size();
                        const size_t evals = vsamples.size();

                        // create workers (try different random branches)
                        for (size_t n = 0; n < n_workers; n ++)
                        {
                                wpool.enqueue([=, &states, &task, &tsamples, &vsamples, &loss, &model, &x0, &mutex]()
                                {
                                        // prepare parameters
                                        stochastic_state_t state;
                                        state.init(model);
                                        state.init(x0, opt_log_gamma, opt_log_lambda);

                                        // stochastic optimization
                                        timer_t timer;
                                        sgd(task, tsamples, vsamples, loss, iters, evals, state);

                                        const thread_pool_t::lock_t lock(mutex);
                                        states.push_back(state);

                                        log_info() << "stochastic trainer: optim ["
                                                   << "param = (" << state.gamma() << ", " << state.lambda()
                                                   << "), loss = " << state.m_value << "/" << state.m_error
                                                   << "] done in " << timer.elapsed() << ".";
                                });
                        }

                        wpool.wait();

                        // select the optimum model
                        std::sort(states.begin(), states.end());
                        const stochastic_state_t& opt_state = states[0];

                        model.load_params(opt_state.m_param);

                        // log
                        log_info() << "stochastic trainer: optim ["
                                   << "param = (" << opt_state.gamma() << ", " << opt_state.lambda()
                                   << "), loss* = " << opt_state.m_value << "/" << opt_state.m_error
                                   << ", iters = " << iters << "/" << evals << "].";
                }

                // OK
                return true;
        }

        /////////////////////////////////////////////////////////////////////////////////////////
}
