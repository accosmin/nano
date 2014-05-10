#include "trainer.h"
#include "common/timer.h"
#include "common/logger.h"
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
                        m_lambda(std::numeric_limits<scalar_t>::max())
        {
        }

        /////////////////////////////////////////////////////////////////////////////////////////

        bool trainer_state_t::update(const vector_t& params,
                    scalar_t tvalue, scalar_t terror,
                    scalar_t vvalue, scalar_t verror,
                    scalar_t lambda)
        {
                if (verror < m_verror)
                {
                        m_params = params;
                        m_tvalue = tvalue;
                        m_terror = terror;
                        m_vvalue = vvalue;
                        m_verror = verror;
                        m_lambda = lambda;
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
                              state.m_lambda);
        }

        /////////////////////////////////////////////////////////////////////////////////////////

        bool trainer_t::train(
                const task_t& task, const sampler_t& tsampler, const sampler_t& vsampler, size_t nthreads,
                const loss_t& loss, const string_t& optimizer, size_t iterations, scalar_t epsilon,
                const string_t& regularizer, const model_t& model, trainer_state_t& state)
        {
                // no regularization
                if (regularizer == "none")
                {
                        accumulator_t ldata(model,
                                            accumulator_t::type::value,
                                            accumulator_t::regularizer::none);
                        accumulator_t gdata(model,
                                            accumulator_t::type::vgrad,
                                            accumulator_t::regularizer::none);

                        trainer_t::train(task, tsampler, vsampler, nthreads,
                                         loss, optimizer, iterations, epsilon,
                                         model.params(), ldata, gdata, state);
                }

                // L2-norm regularization
                else if (regularizer == "l2")
                {
                        const scalars_t lambdas = { 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1.0 };

                        // regularize the loss
                        for (scalar_t lambda : lambdas)
                        {
                                accumulator_t ldata(model,
                                                    accumulator_t::type::value,
                                                    accumulator_t::regularizer::l2norm, lambda);
                                accumulator_t gdata(model,
                                                    accumulator_t::type::vgrad,
                                                    accumulator_t::regularizer::l2norm, lambda);

                                trainer_t::train(task, tsampler, vsampler, nthreads,
                                                 loss, optimizer, iterations, epsilon,
                                                 model.params(), ldata, gdata, state);
                        }
                }

                // variational regularization
                else if (regularizer == "var")
                {
                        const scalars_t lambdas = { 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1.0 };

                        // regularize the loss
                        for (scalar_t lambda : lambdas)
                        {
                                accumulator_t ldata(model,
                                                    accumulator_t::type::value,
                                                    accumulator_t::regularizer::variational, lambda);
                                accumulator_t gdata(model,
                                                    accumulator_t::type::vgrad,
                                                    accumulator_t::regularizer::variational, lambda);

                                trainer_t::train(task, tsampler, vsampler, nthreads,
                                                 loss, optimizer, iterations, epsilon,
                                                 model.params(), ldata, gdata, state);
                        }
                }

                else
                {
                        log_error() << "trainer: invalid regularization method <" << regularizer << ">!";
                        return false;
                }

                // OK
                return true;
        }

        /////////////////////////////////////////////////////////////////////////////////////////

        bool trainer_t::train(
                const task_t& task, const sampler_t& tsampler, const sampler_t& vsampler, size_t nthreads,
                const loss_t& loss, const string_t& optimizer, size_t iterations, scalar_t epsilon,
                const vector_t& x0, accumulator_t& ldata, accumulator_t& gdata, trainer_state_t& state)
        {
                samples_t utsamples = tsampler.get();
                samples_t uvsamples = vsampler.get();

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
                        ldata.update_mt(task, utsamples, loss, nthreads);
                        const scalar_t tvalue = ldata.value();

                        return tvalue;
                };

                auto fn_fval_grad = [&] (const vector_t& x, vector_t& gx)
                {
                        // stochastic mode: resample training & validation samples
                        if (tsampler.is_random())
                        {
                                utsamples = tsampler.get();
                        }
                        if (vsampler.is_random())
                        {
                                uvsamples = vsampler.get();
                        }

                        // training samples: loss value & gradient
                        gdata.reset(x);
                        gdata.update_mt(task, utsamples, loss, nthreads);
                        const scalar_t tvalue = gdata.value();
                        const scalar_t terror = gdata.error();
                        gx = gdata.vgrad();

                        // validation samples: loss value
                        ldata.reset(x);
                        ldata.update_mt(task, uvsamples, loss, nthreads);
                        const scalar_t vvalue = ldata.value();
                        const scalar_t verror = ldata.error();

                        // update the optimum state
                        state.update(x, tvalue, terror, vvalue, verror, ldata.lambda());

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
                        log_info() << "[loss = " << result.f
                                   << ", grad = " << result.g.lpNorm<Eigen::Infinity>()
                                   << ", funs = " << result.n_fval_calls() << "/" << result.n_grad_calls()
                                   << ", train* = " << state.m_tvalue << "/" << state.m_terror
                                   << ", valid* = " << state.m_vvalue << "/" << state.m_verror
                                   << ", lambda* = " << ldata.lambda() << "/" << state.m_lambda
                                   << "] done in " << timer.elapsed() << ".";
                };

                // assembly optimization problem & optimize the model
                const opt_problem_t problem(fn_size, fn_fval, fn_fval_grad);

                const opt_opulog_t fn_ulog_ref = std::bind(fn_ulog, _1, std::ref(timer));

                if (text::iequals(optimizer, "lbfgs"))
                {
                        optimize::lbfgs(problem, x0, iterations, epsilon, fn_wlog, fn_elog, fn_ulog_ref);
                }
                else if (text::iequals(optimizer, "cgd"))
                {
                        optimize::cgd_hs(problem, x0, iterations, epsilon, fn_wlog, fn_elog, fn_ulog_ref);
                }
                else if (text::iequals(optimizer, "gd"))
                {
                        optimize::gd(problem, x0, iterations, epsilon, fn_wlog, fn_elog, fn_ulog_ref);
                }
                else
                {
                        log_error() << "trainer: invalid optimization method <" << optimizer << ">!";
                        return false;
                }

                // OK
                return true;
        }

        /////////////////////////////////////////////////////////////////////////////////////////
}
	
