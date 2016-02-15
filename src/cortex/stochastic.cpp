#include "sampler.h"
#include "stochastic.h"
#include "util/timer.h"
#include "util/logger.h"
#include "accumulator.h"
#include "math/stoch.hpp"
#include "thread/thread.h"
#include "text/to_string.hpp"
#include <tuple>

namespace cortex
{
        static trainer_result_t train(trainer_data_t& data,
                const math::stoch_optimizer optimizer, const size_t epochs, const bool verbose)
        {
                trainer_result_t result;

                const cortex::timer_t timer;

                const auto batch_size = 16 * thread::n_threads();

                // set the sampling size
                data.m_tsampler.push(batch_size);

                // construct the optimization problem
                size_t epoch = 0;
                const size_t epoch_size = data.epoch_size(batch_size);

                auto fn_size = cortex::make_opsize(data);
                auto fn_fval = cortex::make_opfval(data);
                auto fn_grad = cortex::make_opgrad(data);

                auto fn_ulog = [&] (const opt_state_t& state)
                {
                        // evaluate training samples
                        data.m_lacc.set_params(state.x);
                        data.m_tsampler.pop();                  //
                        data.m_lacc.update(data.m_task, data.m_tsampler.get(), data.m_loss);
                        data.m_tsampler.push(batch_size);       //
                        const scalar_t tvalue = data.m_lacc.value();
                        const scalar_t terror_avg = data.m_lacc.avg_error();
                        const scalar_t terror_var = data.m_lacc.var_error();

                        // evaluate validation samples
                        data.m_lacc.set_params(state.x);
                        data.m_lacc.update(data.m_task, data.m_vsampler.get(), data.m_loss);
                        const scalar_t vvalue = data.m_lacc.value();
                        const scalar_t verror_avg = data.m_lacc.avg_error();
                        const scalar_t verror_var = data.m_lacc.var_error();

                        // OK, update the optimum solution
                        const auto milis = timer.milliseconds();
                        const auto ret = result.update(state.x,
                                {milis, ++ epoch, tvalue, terror_avg, terror_var, vvalue, verror_avg, verror_var},
                                {data.lambda()});

                        if (verbose)
                        log_info()
                                << "[train = " << tvalue << "/" << terror_avg
                                << ", valid = " << vvalue << "/" << verror_avg
                                << " (" << text::to_string(ret) << ")"
                                << ", epoch = " << epoch << "/" << epochs
                                << ", batch = " << batch_size
                                << ", lambda = " << data.lambda()
                                << "] done in " << timer.elapsed() << ".";

                        return !cortex::is_done(ret);
                };

                // Optimize the model
                math::minimize(opt_problem_t(fn_size, fn_fval, fn_grad), fn_ulog,
                               data.m_x0, optimizer, epochs, epoch_size);

                // revert to the original sampler
                data.m_tsampler.pop();

                return result;
        }

        trainer_result_t stochastic_train(
                const model_t& model,
                const task_t& task, const sampler_t& tsampler, const sampler_t& vsampler, size_t nthreads,
                const loss_t& loss, const criterion_t& criterion,
                math::stoch_optimizer optimizer, size_t epochs, bool verbose)
        {
                vector_t x0;
                model.save_params(x0);

                // setup accumulators
                accumulator_t lacc(model, criterion, criterion_t::type::value); lacc.set_threads(nthreads);
                accumulator_t gacc(model, criterion, criterion_t::type::vgrad); gacc.set_threads(nthreads);

                trainer_data_t data(task, tsampler, vsampler, loss, x0, lacc, gacc);

                // tune the regularization factor (if needed)
                const auto op = [&] (scalar_t lambda)
                {
                        data.set_lambda(lambda);
                        return train(data, optimizer, epochs, verbose);
                };

                if (data.m_lacc.can_regularize())
                {
                        return std::get<0>(math::tune_fixed(op, cortex::tunable_lambdas()));
                }
                else
                {
                        return op(0.0);
                }
        }
}
