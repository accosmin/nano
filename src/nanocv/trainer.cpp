#include "trainer.h"
#include "common/timer.h"
#include "common/logger.h"
#include "common/thread_pool.h"
#include "optimize/opt_gd.hpp"
#include "optimize/opt_cgd.hpp"
#include "optimize/opt_lbfgs.hpp"
#include "sampler.h"
#include "accumulator.h"

namespace ncv
{
        /////////////////////////////////////////////////////////////////////////////////////////

        trainer_state_t::trainer_state_t(size_t n_parameters)
                :       m_params(n_parameters),
                        m_tvalue(std::numeric_limits<scalar_t>::max()),
                        m_terror(std::numeric_limits<scalar_t>::max()),
                        m_vvalue(std::numeric_limits<scalar_t>::max()),
                        m_verror(std::numeric_limits<scalar_t>::max()),
                        m_lambda(std::numeric_limits<scalar_t>::max()),
                        m_fcalls(0),
                        m_gcalls(0)
        {
        }

        /////////////////////////////////////////////////////////////////////////////////////////

        bool trainer_state_t::update(const vector_t& params,
                    scalar_t tvalue, scalar_t terror,
                    scalar_t vvalue, scalar_t verror,
                    scalar_t lambda, size_t fcalls, size_t gcalls)
        {
                if (verror < m_verror)
                {
                        m_params = params;
                        m_tvalue = tvalue;
                        m_terror = terror;
                        m_vvalue = vvalue;
                        m_verror = verror;
                        m_lambda = lambda;
                        m_fcalls = fcalls;
                        m_gcalls = gcalls;
                        return true;
                }

                else
                {
                        return false;
                }
        }

        /////////////////////////////////////////////////////////////////////////////////////////

        bool trainer_state_t::update(const trainer_state_t& state)
        {
                return update(state.m_params,
                              state.m_tvalue, state.m_terror, state.m_vvalue, state.m_verror,
                              state.m_lambda, state.m_fcalls, state.m_gcalls);
        }

        /////////////////////////////////////////////////////////////////////////////////////////

        opt_state_t batch_train(
                const task_t& task, const samples_t& tsamples, const samples_t& vsamples, size_t nthreads,
                const loss_t& loss, batch_optimizer optimizer, size_t epochs, size_t iterations, scalar_t epsilon,
                const vector_t& x0, accumulator_t& ldata, accumulator_t& gdata, trainer_state_t& state)
        {
                size_t iteration = 0;

                // construct the optimization problem
                const timer_t timer;

                auto fn_size = [&] ()
                {
                        return ldata.dimensions();
                };

                auto fn_fval = [&] (const vector_t& x)
                {
                        // training samples: loss value
                        ldata.reset(x);
                        ldata.update(task, tsamples, loss, nthreads);
                        const scalar_t tvalue = ldata.value();

                        return tvalue;
                };

                auto fn_fval_grad = [&] (const vector_t& x, vector_t& gx)
                {
                        // training samples: loss value & gradient
                        gdata.reset(x);
                        gdata.update(task, tsamples, loss, nthreads);
                        const scalar_t tvalue = gdata.value();
                        gx = gdata.vgrad();

                        return tvalue;
                };

                auto fn_wlog = [] (const string_t& message)
                {
                        log_warning() << message;
                };
                auto fn_elog = [] (const string_t& message)
                {
                        log_error() << message;
                };
                auto fn_ulog = [&] (const opt_state_t& result, const timer_t& timer)
                {
                        ++ iteration;

                        if ((iteration % iterations) == 0)
                        {
                                const scalar_t tvalue = gdata.value();
                                const scalar_t terror = gdata.error();

                                // validation samples: loss value
                                ldata.reset(result.x);
                                ldata.update(task, vsamples, loss, nthreads);
                                const scalar_t vvalue = ldata.value();
                                const scalar_t verror = ldata.error();

                                // update the optimum state
                                state.update(result.x, tvalue, terror, vvalue, verror,
                                             ldata.lambda(), result.n_fval_calls(), result.n_grad_calls());

                                log_info() << "[train = " << tvalue << "/" << terror
                                           << ", valid = " << vvalue << "/" << verror
                                           << ", grad = " << result.g.lpNorm<Eigen::Infinity>()
                                           << ", lambda = " << ldata.lambda()
                                           << ", funs = " << result.n_fval_calls() << "/" << result.n_grad_calls()
                                           << "] done in " << timer.elapsed() << ".";
                        }
                };

                // assembly optimization problem & optimize the model
                const opt_problem_t problem(fn_size, fn_fval, fn_fval_grad);

                const opt_opulog_t fn_ulog_ref = std::bind(fn_ulog, _1, std::ref(timer));

                switch (optimizer)
                {
                case batch_optimizer::LBFGS:
                        return optimize::lbfgs(problem, x0, epochs * iterations, epsilon,
                                               fn_wlog, fn_elog, fn_ulog_ref);

                case batch_optimizer::CGD:
                        return optimize::cgd_hs(problem, x0, epochs * iterations, epsilon,
                                                fn_wlog, fn_elog, fn_ulog_ref);

                case batch_optimizer::GD:
                default:
                        return optimize::gd(problem, x0, epochs * iterations, epsilon,
                                            fn_wlog, fn_elog, fn_ulog_ref);
                }
        }

        /////////////////////////////////////////////////////////////////////////////////////////

        static void stochastic_train(
                const task_t& task, samples_t& tsamples, const samples_t& vsamples, const loss_t& loss,
                size_t epochs, scalar_t alpha0, scalar_t beta, stochastic_optimizer type, scalar_t lambda,
                const model_t& model, trainer_state_t& state, thread_pool_t::mutex_t& mutex)
        {
                accumulator_t ldata(model, accumulator_t::type::value, lambda);
                accumulator_t gdata(model, accumulator_t::type::vgrad, lambda);

                random_t<size_t> xrng(0, tsamples.size());
                rnd_t xrnd(xrng);

                // (weighted-average) stochastic gradient descent
                timer_t timer;

                vector_t x = model.params();
                vector_t xparam = x;

                vector_t xavg = x;
                vector_t gavg(x.size());
                gavg.setZero();

                scalar_t alpha = alpha0;
                scalar_t sumb = 1.0 / alpha;

                for (size_t e = 0; e < epochs; e ++)
                {
                        std::random_shuffle(tsamples.begin(), tsamples.end(), xrnd);

                        switch (type)
                        {
                        case stochastic_optimizer::SG:
                                for (size_t i = 0; i < tsamples.size(); i ++, alpha *= beta)
                                {
                                        gdata.reset(x);
                                        gdata.update(task, tsamples[i], loss);

                                        x.noalias() -= alpha * gdata.vgrad();
                                }
                                xparam = x;
                                break;

                        case stochastic_optimizer::SGA:
                                for (size_t i = 0; i < tsamples.size(); i ++, alpha *= beta)
                                {
                                        gdata.reset(x);
                                        gdata.update(task, tsamples[i], loss);

                                        const vector_t g = gdata.vgrad();

                                        const scalar_t b = 1.0 / alpha;
                                        gavg = (gavg * sumb + g * b) / (sumb + b);
                                        sumb = sumb + b;

                                        x.noalias() -= alpha * gavg;
                                }
                                xparam = xavg;
                                break;

                        case stochastic_optimizer::SIA:
                        default:
                                for (size_t i = 0; i < tsamples.size(); i ++, alpha *= beta)
                                {
                                        gdata.reset(x);
                                        gdata.update(task, tsamples[i], loss);

                                        x.noalias() -= alpha * gdata.vgrad();

                                        const scalar_t b = 1.0 / alpha;
                                        xavg = (xavg * sumb + x * b) / (sumb + b);
                                        sumb = sumb + b;
                                }
                                xparam = x;
                                break;
                        }

                        // evaluate training samples
                        ldata.reset(xparam);
                        ldata.update(task, tsamples, loss);
                        const scalar_t tvalue = ldata.value();
                        const scalar_t terror = ldata.error();

                        // evaluate validation samples
                        ldata.reset(xparam);
                        ldata.update(task, vsamples, loss);
                        const scalar_t vvalue = ldata.value();
                        const scalar_t verror = ldata.error();

                        // OK, update the optimum solution
                        const thread_pool_t::lock_t lock(mutex);

                        if (state.update(xparam, tvalue, terror, vvalue, verror,
                                         ldata.lambda(), e * tsamples.size(), e * tsamples.size()))
                        {
                                log_info()
                                << "[rate = " << alpha << "/" << alpha0
                                << ", epoch = " << e << "/" << epochs
                                << ", train* = " << state.m_tvalue << "/" << state.m_terror
                                << ", valid* = " << state.m_vvalue << "/" << state.m_verror
                                << ", lambda* = " << ldata.lambda() << "/" << state.m_lambda
                                << "] done in " << timer.elapsed() << ".";
                        }
                }
        }

        /////////////////////////////////////////////////////////////////////////////////////////

        opt_state_t stochastic_train(
                const task_t& task, const samples_t& tsamples, const samples_t& vsamples, size_t nthreads,
                const loss_t& loss, stochastic_optimizer optimizer, size_t epochs, size_t iterations, scalar_t epsilon,
                const vector_t& x0, accumulator_t& ldata, accumulator_t& gdata, trainer_state_t& state)
        {
                // prepare workers
                thread_pool_t wpool(nthreads);
                thread_pool_t::mutex_t mutex;

                // tune the learning rate
                const scalar_t max_alpha = 1e-1;
                const scalar_t min_alpha = 1e-3;
                const scalar_t var_alpha = std::exp((std::log(max_alpha) - std::log(min_alpha))
                                           / std::min(size_t(8), wpool.n_workers()));

                for (scalar_t alpha0 = min_alpha; alpha0 <= max_alpha; alpha0 *= var_alpha)
                {
                        wpool.enqueue([=, &model, &task, &tsamples, &vsamples, &loss, &state, &mutex]()
                        {
                                stochastic_train(task, tsamples, vsamples, loss,
                                                 epochs, alpha0, beta, optimizer, lambda,
                                                 x0, ldata, gdata, state, mutex);
                        });
                }

                wpool.wait();

                opt_state_t result;
                result;
        }

        /////////////////////////////////////////////////////////////////////////////////////////
}
	
