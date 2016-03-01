#include "batch.h"
#include "sampler.h"
#include "util/timer.h"
#include "util/logger.h"
#include "accumulator.h"
#include "math/tune.hpp"
#include "math/batch.hpp"
#include "text/to_string.hpp"

namespace cortex
{
        static opt_state_t train_batch(
                trainer_data_t& data,
                math::batch_optimizer optimizer, size_t iterations, scalar_t epsilon,
                timer_t& timer, trainer_result_t& result, bool verbose)
        {
                size_t iteration = 0;

                // construct the optimization problem
                auto fn_size = cortex::make_opsize(data);
                auto fn_fval = cortex::make_opfval(data);
                auto fn_grad = cortex::make_opgrad(data);

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
                        const auto milis = timer.milliseconds();
                        const auto ret = result.update(state.x,
                                {milis, ++ iteration, tvalue, terror_avg, terror_var, vvalue, verror_avg, verror_var},
                                {data.lambda()});

                        if (verbose)
                        log_info()
                                << "[train = " << tvalue << "/" << terror_avg
                                << ", valid = " << vvalue << "/" << verror_avg
                                << " (" << text::to_string(ret) << ")"
                                << ", epoch = " << iteration << "/" << iterations
                                << ", lambda = " << data.lambda()
                                << ", calls = " << state.m_fcalls << "/" << state.m_gcalls
                                << "] done in " << timer.elapsed() << ".";

                        return !cortex::is_done(ret);
                };

                // assembly optimization problem & optimize the model
                return math::minimize(opt_problem_t(fn_size, fn_fval, fn_grad), fn_ulog,
                                     data.m_x0, optimizer, iterations, epsilon);
        }

        trainer_result_t batch_train(
                const model_t& model, const task_t& task, const sampler_t& tsampler, const sampler_t& vsampler, size_t nthreads,
                const loss_t& loss, const criterion_t& criterion,
                math::batch_optimizer optimizer, size_t iterations, scalar_t epsilon,
                bool verbose)
        {
                vector_t x0;
                model.save_params(x0);

                // setup acumulators
                accumulator_t lacc(model, criterion, criterion_t::type::value); lacc.set_threads(nthreads);
                accumulator_t gacc(model, criterion, criterion_t::type::vgrad); gacc.set_threads(nthreads);

                trainer_data_t data(task, tsampler, vsampler, loss, x0, lacc, gacc);

                // tune the regularization factor (if needed)
                const auto op = [&] (scalar_t lambda)
                {
                        data.set_lambda(lambda);

                        trainer_result_t result;
                        timer_t timer;

                        train_batch(data, optimizer, iterations, epsilon, timer, result, verbose);

                        return result;
                };

                if (data.m_lacc.can_regularize())
                {
                        const auto space = math::make_log10_space(-6.0, +6.0, 0.5);
                        return math::tune(op, space).optimum();
                }
                else
                {
                        return op(0.0);
                }
        }
}

