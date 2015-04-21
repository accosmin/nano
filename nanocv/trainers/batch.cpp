#include "batch.h"
#include "nanocv/timer.h"
#include "nanocv/logger.h"
#include "nanocv/sampler.h"
#include "nanocv/minimize.h"
#include "nanocv/accumulator.h"
#include "nanocv/log_search.hpp"

namespace ncv
{
        namespace detail
        {        
                static opt_state_t batch_train(
                        trainer_data_t& data,
                        optim::batch_optimizer optimizer, size_t iterations, scalar_t epsilon,
                        timer_t& timer, trainer_result_t& result, bool verbose)
                {
                        size_t iteration = 0;

                        // construct the optimization problem
                        auto fn_size = ncv::make_opsize(data);
                        auto fn_fval = ncv::make_opfval(data);
                        auto fn_grad = ncv::make_opgrad(data);

                        auto fn_wlog = verbose ? ncv::make_opwlog() : nullptr;
                        auto fn_elog = verbose ? ncv::make_opelog() : nullptr;
                        auto fn_ulog = [&] (const opt_state_t& state)
                        {
                                const scalar_t tvalue = data.m_gacc.value();
                                const scalar_t terror_avg = data.m_gacc.avg_error();
                                const scalar_t terror_var = data.m_gacc.var_error();

                                // validation samples: loss value
                                data.m_lacc.set_params(state.x);
                                data.m_lacc.update(data.m_task, data.m_vsampler.get(), data.m_loss);
                                const scalar_t vvalue = data.m_lacc.value();
                                const scalar_t verror_avg = data.m_lacc.avg_error();
                                const scalar_t verror_var = data.m_lacc.var_error();

                                // update the optimum state
                                const auto ret = result.update(
                                        state.x, tvalue, terror_avg, terror_var, vvalue, verror_avg, verror_var,
                                        ++ iteration, scalars_t({ data.lambda() }));

                                if (verbose)
                                log_info()
                                        << "[train = " << tvalue << "/" << terror_avg
                                        << ", valid = " << vvalue << "/" << verror_avg
                                        << " (" << text::to_string(ret) << ")"
                                        << ", xnorm = " << state.x.lpNorm<Eigen::Infinity>()
                                        << ", gnorm = " << state.g.lpNorm<Eigen::Infinity>()
                                        << ", epoch = " << iteration
                                        << ", lambda = " << data.lambda()
                                        << ", calls = " << state.n_fval_calls() << "/" << state.n_grad_calls()
                                        << "] done in " << timer.elapsed() << ".";

                                return ret != trainer_result_update_code_t::overfitting;
                        };

                        // assembly optimization problem & optimize the model
                        return ncv::minimize(fn_size, fn_fval, fn_grad, fn_wlog, fn_elog, fn_ulog,
                                             data.m_x0, optimizer, iterations, epsilon);
                }
        }
        
        trainer_result_t batch_train(
                const model_t& model, const task_t& task, const sampler_t& tsampler, const sampler_t& vsampler, size_t nthreads,
                const loss_t& loss, const string_t& criterion, 
                optim::batch_optimizer optimizer, size_t iterations, scalar_t epsilon,
                bool verbose)
        {
                vector_t x0;
                model.save_params(x0);

                // operator to train for a given regularization factor
                const auto op = [&] (scalar_t lambda)
                {
                        trainer_result_t result;
                        timer_t timer;

                        // optimize the model
                        accumulator_t lacc(model, nthreads, criterion, criterion_t::type::value, lambda);
                        accumulator_t gacc(model, nthreads, criterion, criterion_t::type::vgrad, lambda);

                        trainer_data_t data(task, tsampler, vsampler, loss, x0, lacc, gacc);

                        detail::batch_train(data, optimizer, iterations, epsilon, timer, result, verbose);

                        // OK
                        return result;
                };

                // tune the regularization factor (if needed)
                if (accumulator_t::can_regularize(criterion))
                {
                        return log10_min_search(op, -6.0, +0.0, 0.2, 4).first;
                }

                else
                {
                        return op(0.0);
                }
        }
}
	
