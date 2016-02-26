#include "sampler.h"
#include "minibatch.h"
#include "util/timer.h"
#include "util/logger.h"
#include "accumulator.h"
#include "math/batch.hpp"
#include "thread/thread.h"
#include "text/to_string.hpp"
#include "math/tune_fixed.hpp"

namespace cortex
{
        template
        <
                typename toperator
        >
        static void train_epoch(trainer_data_t& data,
                const size_t epoch_size, const size_t batch_size, const toperator& op)
        {
                auto& tsampler = data.m_tsampler;

                for (size_t i = 0; i < epoch_size; ++ i)
                {
                        tsampler.push(batch_size);
                        tsampler.push(tsampler.get());

                        op();

                        tsampler.pop();
                        tsampler.pop();
                }
        }

        static trainer_result_t train(trainer_data_t& data,
                math::batch_optimizer optimizer, const size_t epochs, const scalar_t epsilon, const bool verbose)
        {
                const cortex::timer_t timer;

                trainer_result_t result;

                const auto batch_size = 32 * thread::n_threads();
                const auto epoch_size = data.epoch_size(batch_size);
                const auto epoch_iterations = size_t(4);
                const auto history_size = epoch_iterations;

                // construct the optimization problem
                const auto fn_size = cortex::make_opsize(data);
                const auto fn_fval = cortex::make_opfval(data);
                const auto fn_grad = cortex::make_opgrad(data);

                const auto fn_ulog = nullptr;

                // optimize the model
                vector_t x = data.m_x0;

                for (size_t epoch = 1; epoch <= epochs; ++ epoch)
                {
                        train_epoch(data, epoch_size, batch_size, [&] ()
                        {
                                const opt_state_t state = math::minimize(
                                        opt_problem_t(fn_size, fn_fval, fn_grad), fn_ulog,
                                        x, optimizer, epoch_iterations, epsilon, history_size);

                                x = state.x;
                        });

                        // training samples: loss value
                        data.m_lacc.set_params(x);
                        data.m_lacc.update(data.m_task, data.m_tsampler.get(), data.m_loss);
                        const scalar_t tvalue = data.m_lacc.value();
                        const scalar_t terror_avg = data.m_lacc.avg_error();
                        const scalar_t terror_var = data.m_lacc.var_error();

                        // validation samples: loss value
                        data.m_lacc.set_params(x);
                        data.m_lacc.update(data.m_task, data.m_vsampler.get(), data.m_loss);
                        const scalar_t vvalue = data.m_lacc.value();
                        const scalar_t verror_avg = data.m_lacc.avg_error();
                        const scalar_t verror_var = data.m_lacc.var_error();

                        // update the optimum state
                        const auto milis = timer.milliseconds();
                        const auto ret = result.update(x,
                                {milis, epoch, tvalue, terror_avg, terror_var, vvalue, verror_avg, verror_var},
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

                        if (cortex::is_done(ret))
                        {
                                break;
                        }
                }

                return result;
        }

        trainer_result_t minibatch_train(
                const model_t& model,
                const task_t& task, const sampler_t& tsampler, const sampler_t& vsampler, size_t nthreads,
                const loss_t& loss, const criterion_t& criterion,
                math::batch_optimizer optimizer, size_t epochs, scalar_t epsilon, bool verbose)
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
                        return train(data, optimizer, epochs, epsilon, verbose);
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

