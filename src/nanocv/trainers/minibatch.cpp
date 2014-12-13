#include "batch.h"
#include "accumulator.h"
#include "sampler.h"
#include "io/logger.h"
#include "common/log_search.hpp"
#include "common/timer.h"
#include "optimize/batch_gd.hpp"
#include "optimize/batch_cgd.hpp"
#include "optimize/batch_lbfgs.hpp"

namespace ncv
{
        namespace detail
        {        
                static opt_state_t minibatch_train(
                        trainer_data_t& data,
                        batch_optimizer optimizer, size_t iterations, scalar_t epsilon, size_t epoch,
                        timer_t& timer, trainer_result_t& result)
                {
                        size_t iteration = 0;  
                        
                        const samples_t tsamples = data.m_tsampler.get();
                        const samples_t vsamples = data.m_vsampler.get();

                        // construct the optimization problem
                        auto fn_size = [&] ()
                        {
                                return data.m_lacc.psize();
                        };

                        auto fn_fval = [&] (const vector_t& x)
                        {
                                // training samples: loss value
                                data.m_lacc.reset(x);
                                data.m_lacc.update(data.m_task, tsamples, data.m_loss);
                                const scalar_t tvalue = data.m_lacc.value();

                                return tvalue;
                        };

                        auto fn_fval_grad = [&] (const vector_t& x, vector_t& gx)
                        {
                                // training samples: loss value & gradient
                                data.m_gacc.reset(x);
                                data.m_gacc.update(data.m_task, tsamples, data.m_loss);
                                const scalar_t tvalue = data.m_gacc.value();
                                gx = data.m_gacc.vgrad();

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
                                if ((++ iteration) == iterations)
                                {
                                        const scalar_t tvalue = data.m_gacc.value();
                                        const scalar_t terror = data.m_gacc.error();

                                        // validation samples: loss value
                                        data.m_lacc.reset(state.x);
                                        data.m_lacc.update(data.m_task, vsamples, data.m_loss);
                                        const scalar_t vvalue = data.m_lacc.value();
                                        const scalar_t verror = data.m_lacc.error();

                                        // update the optimum state
                                        result.update(state.x, tvalue, terror, vvalue, verror,
                                                      epoch, scalars_t({ data.m_lacc.lambda() }));

                                        log_info()
                                                << "[train = " << tvalue << "/" << terror << "/=" << tsamples.size()
                                                << ", valid = " << vvalue << "/" << verror << "/=" << vsamples.size()
                                                << ", param = " << state.x.lpNorm<Eigen::Infinity>()
                                                << ", epoch = " << epoch
                                                << ", lambda = " << data.m_lacc.lambda()
                                                << ", calls = " << state.n_fval_calls() << "/" << state.n_grad_calls()
                                                << "] done in " << timer.elapsed() << ".";
                                }
                        };

                        // assembly optimization problem & optimize the model
                        const opt_problem_t problem(fn_size, fn_fval, fn_fval_grad);

                        const size_t history_size = 6;

                        switch (optimizer)
                        {
                        case batch_optimizer::LBFGS:
                                return  optimize::batch_lbfgs<opt_problem_t>
                                        (iterations, epsilon, history_size, fn_wlog, fn_elog, fn_ulog)
                                        (problem, data.m_x0);

                        case batch_optimizer::CGD:
                                return  optimize::batch_cgd_pr<opt_problem_t>
                                        (iterations, epsilon, fn_wlog, fn_elog, fn_ulog)
                                        (problem, data.m_x0);

                        case batch_optimizer::GD:
                        default:
                                return  optimize::batch_gd<opt_problem_t>
                                        (iterations, epsilon, fn_wlog, fn_elog, fn_ulog)
                                        (problem, data.m_x0);
                        }
                }
        }

        trainer_result_t minibatch_train(
                const model_t& model, const task_t& task, sampler_t& tsampler, sampler_t& vsampler, size_t nthreads,
                const loss_t& loss, const string_t& criterion, size_t batchsize, scalar_t batchratio,
                batch_optimizer optimizer, size_t epochs, size_t iterations, scalar_t epsilon)
        {
                // operator to train for a given regularization factor
                const auto op = [&] (scalar_t lambda)
                {
                        trainer_result_t result;
                        timer_t timer;

                        // optimize the model
                        vector_t x0;
                        model.save_params(x0);

                        for (size_t epoch = 1, tsize = batchsize; epoch <= epochs;
                                epoch ++, tsize = static_cast<size_t>(tsize * batchratio))
                        {
                                // random subset of training samples
                                if (tsize < tsampler.size())
                                {
                                        tsampler.setup(sampler_t::stype::uniform, tsize);
                                }
                                else
                                {
                                        tsampler.setup(sampler_t::stype::batch);
                                }

                                // all validation samples
                                vsampler.setup(sampler_t::stype::batch);

                                accumulator_t lacc(model, nthreads, criterion, criterion_t::type::value, lambda);
                                accumulator_t gacc(model, nthreads, criterion, criterion_t::type::vgrad, lambda);

                                trainer_data_t data(task, tsampler, vsampler, loss, x0, lacc, gacc);

                                const opt_state_t state =
                                detail::minibatch_train(data, optimizer, iterations, epsilon, epoch, timer, result);
                                x0 = state.x;

                                // NB: this will cause resampling of the training data!
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
	
