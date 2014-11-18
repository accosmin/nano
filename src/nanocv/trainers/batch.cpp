#include "batch.h"
#include "accumulator.h"
#include "sampler.h"
#include "io/logger.h"
#include "common/log_search.hpp"
#include "common/timer.h"
#include "optimize/opt_gd.hpp"
#include "optimize/opt_cgd.hpp"
#include "optimize/opt_lbfgs.hpp"

namespace ncv
{
        namespace detail
        {        
                static opt_state_t batch_train(
                        trainer_data_t& data, 
                        batch_optimizer optimizer, size_t epochs, size_t iterations, scalar_t epsilon, size_t& epoch,
                        trainer_result_t& result)
                {
                        size_t iteration = 0;  
                        
                        samples_t tsamples = data.m_tsampler.get();
                        samples_t vsamples = data.m_vsampler.get();

                        // construct the optimization problem
                        const timer_t timer;

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
                        auto fn_ulog = [&] (const opt_state_t& state, const timer_t& timer)
                        {
                                ++ iteration;
                                if ((iteration % iterations) == 0)
                                {
                                        ++ epoch;
                                        
                                        const scalar_t tvalue = data.m_gacc.value();
                                        const scalar_t terror = data.m_gacc.error();

                                        // validation samples: loss value
                                        data.m_lacc.reset(state.x);
                                        data.m_lacc.update(data.m_task, vsamples, data.m_loss);
                                        const scalar_t vvalue = data.m_lacc.value();
                                        const scalar_t verror = data.m_lacc.error();

                                        // update the optimum state
                                        result.update(state.x, tvalue, terror, vvalue, verror, epoch,
                                                      scalars_t({ data.m_lacc.lambda() }));

                                        log_info()
                                                << "[train = " << tvalue << "/" << terror
                                                << ", valid = " << vvalue << "/" << verror
                                                << ", param = " << state.x.lpNorm<Eigen::Infinity>()
                                                << ", calls = " << state.n_fval_calls() << "/" << state.n_grad_calls()
                                                << ", lambda = " << data.m_lacc.lambda()
                                                << "] done in " << timer.elapsed() << ".";
                                                
                                        // resample
                                        if (iteration < iterations)
                                        {
                                                tsamples = data.m_tsampler.get();
                                                vsamples = data.m_vsampler.get();  
                                        }
                                }
                        };

                        // assembly optimization problem & optimize the model
                        const opt_problem_t problem(fn_size, fn_fval, fn_fval_grad);

                        const opt_opulog_t fn_ulog_ref = std::bind(fn_ulog, _1, std::ref(timer));

                        switch (optimizer)
                        {
                        case batch_optimizer::LBFGS:
                                return optimize::lbfgs(problem, data.m_x0, epochs * iterations, epsilon,
                                                       fn_wlog, fn_elog, fn_ulog_ref);

                        case batch_optimizer::CGD:
                                return optimize::cgd_pr(problem, data.m_x0, epochs * iterations, epsilon,
                                                        fn_wlog, fn_elog, fn_ulog_ref);

                        case batch_optimizer::GD:
                        default:
                                return optimize::gd(problem, data.m_x0, epochs * iterations, epsilon,
                                                    fn_wlog, fn_elog, fn_ulog_ref);
                        }
                }
        }
        
        trainer_result_t batch_train(
                const model_t& model, const task_t& task, const sampler_t& tsampler, const sampler_t& vsampler, size_t nthreads,
                const loss_t& loss, const string_t& criterion, 
                batch_optimizer optimizer, size_t cycles, size_t epochs, size_t iterations, scalar_t epsilon)
        {
                // operator to train for a given regularization factor
                const auto op = [&] (scalar_t lambda)
                {
                        accumulator_t lacc(model, nthreads, criterion, criterion_t::type::value, lambda);
                        accumulator_t gacc(model, nthreads, criterion, criterion_t::type::vgrad, lambda);

                        trainer_result_t result;

                        // optimize the model
                        vector_t x;
                        model.save_params(x);
                        for (size_t c = 0, epoch = 0; c < cycles; c ++)
                        {
                                trainer_data_t data(task, tsampler, vsampler, loss, x, lacc, gacc);

                                const opt_state_t state = detail::batch_train(
                                        data, optimizer, epochs, iterations, epsilon, epoch, result);

                                x = state.x;
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
	
