#include "batch_trainer.h"
#include "io/logger.h"
#include "common/math.hpp"
#include "common/log_search.hpp"
#include "common/timer.h"
#include "optimize/batch_gd.hpp"
#include "optimize/batch_cgd.hpp"
#include "optimize/batch_lbfgs.hpp"
#include "accumulator.h"
#include "sampler.h"

namespace ncv
{
        batch_trainer_t::batch_trainer_t(const string_t& parameters)
                :       trainer_t(parameters)
        {
        }

        trainer_result_t batch_trainer_t::train(
                const task_t& task, const fold_t& fold, const loss_t& loss, size_t nthreads, const string_t& criterion,
                model_t& model) const
        {
                if (fold.second != protocol::train)
                {
                        log_error() << "batch trainer: can only train models with training samples!";
                        return trainer_result_t();
                }

                // initialize the model
                model.resize(task, true);
                model.random_params();

                // prune training & validation data
                sampler_t tsampler(task);
                tsampler.setup(fold).setup(sampler_t::atype::annotated);

                sampler_t vsampler(task);
                tsampler.split(80, vsampler);

                if (tsampler.empty() || vsampler.empty())
                {
                        log_error() << "batch trainer: no annotated training samples!";
                        return trainer_result_t();
                }

                // train the model
                const trainer_result_t result = batch_train(task, tsampler, vsampler, nthreads, loss, criterion, model);

                log_info() << "optimum [train = " << result.m_opt_state.m_tvalue << "/" << result.m_opt_state.m_terror
                           << ", valid = " << result.m_opt_state.m_vvalue << "/" << result.m_opt_state.m_verror
                           << ", epoch = " << result.m_opt_epoch
                           << ", config = " << text::concatenate(result.m_opt_config, "/")
                           << "].";

                // OK
                if (result.valid())
                {
                        model.load_params(result.m_opt_params);
                }
                return result;
        }

        trainer_result_t batch_trainer_t::batch_train(
                const task_t& task, const sampler_t& tsampler, const sampler_t& vsampler, size_t nthreads,
                const loss_t& loss, const string_t& criterion, const model_t& model) const
        {
                // parameters
                const size_t iterations = math::clamp(text::from_params<size_t>(configuration(), "iters", 1024), 4, 4096);
                const scalar_t epsilon = math::clamp(text::from_params<scalar_t>(configuration(), "eps", 1e-6), 1e-8, 1e-3);

                const batch_optimizer optimizer = text::from_string<batch_optimizer>
                                (text::from_params<string_t>(configuration(), "opt", "lbfgs"));

                // training & validation samples
                const samples_t tsamples = tsampler.get();
                const samples_t vsamples = vsampler.get();

                vector_t x0;
                model.save_params(x0);

                // operator to train for a given regularization factor
                const auto op = [&] (scalar_t lambda)
                {
                        accumulator_t lacc(model, nthreads, criterion, criterion_t::type::value, lambda);
                        accumulator_t gacc(model, nthreads, criterion, criterion_t::type::vgrad, lambda);

                        trainer_result_t result;
                        size_t iteration = 0;

                        const ncv::timer_t timer;

                        // construct the optimization problem

                        auto fn_size = [&] ()
                        {
                                return lacc.psize();
                        };

                        auto fn_fval = [&] (const vector_t& x)
                        {
                                // training samples: loss value
                                lacc.reset(x);
                                lacc.update(task, tsamples, loss);
                                const scalar_t tvalue = lacc.value();

                                return tvalue;
                        };

                        auto fn_fval_grad = [&] (const vector_t& x, vector_t& gx)
                        {
                                // training samples: loss value & gradient
                                gacc.reset(x);
                                gacc.update(task, tsamples, loss);
                                const scalar_t tvalue = gacc.value();
                                gx = gacc.vgrad();

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
                        const opt_opulog_t fn_ulog = [&] (const opt_state_t& state)
                        {
                                ++ iteration;

                                const scalar_t tvalue = gacc.value();
                                const scalar_t terror = gacc.error();

                                // validation samples: loss value
                                lacc.reset(state.x);
                                lacc.update(task, vsamples, loss);
                                const scalar_t vvalue = lacc.value();
                                const scalar_t verror = lacc.error();

                                // update the optimum state
                                result.update(state.x, tvalue, terror, vvalue, verror, iteration,
                                              scalars_t({ lacc.lambda() }));

                                log_info()
                                        << "[train = " << tvalue << "/" << terror
                                        << ", valid = " << vvalue << "/" << verror
                                        << ", param = " << state.x.lpNorm<Eigen::Infinity>()
                                        << ", calls = " << state.n_fval_calls() << "/" << state.n_grad_calls()
                                        << ", lambda = " << lacc.lambda()
                                        << "] done in " << timer.elapsed() << ".";
                        };

                        // assembly optimization problem & optimize the model
                        const opt_problem_t problem(fn_size, fn_fval, fn_fval_grad);

                        const size_t history_size = 8;

                        switch (optimizer)
                        {
                        case batch_optimizer::LBFGS:
                                optimize::batch_lbfgs<opt_problem_t>
                                        (iterations, epsilon, history_size, fn_wlog, fn_elog, fn_ulog)
                                        (problem, x0);
                                break;

                        case batch_optimizer::CGD:
                                optimize::batch_cgd_pr<opt_problem_t>
                                        (iterations, epsilon, fn_wlog, fn_elog, fn_ulog)
                                        (problem, x0);
                                break;

                        case batch_optimizer::GD:
                        default:
                                optimize::batch_gd<opt_problem_t>
                                        (iterations, epsilon, fn_wlog, fn_elog, fn_ulog)
                                        (problem, x0);
                                break;
                        }

                        // OK
                        return result;
                };

                // tune the regularization factor (if needed)
                if (accumulator_t::can_regularize(criterion))
                {
                        return log_min_search(op, -2.0, +6.0, 0.5, 4);
                }

                else
                {
                        return op(0.0);
                }
        }
}
