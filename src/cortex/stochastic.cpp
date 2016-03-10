#include "sampler.h"
#include "stochastic.h"
#include "util/timer.h"
#include "util/logger.h"
#include "accumulator.h"
#include "math/tune.hpp"
#include "math/stoch.hpp"
#include "thread/thread.h"
#include "text/to_string.hpp"

namespace zob
{
        static trainer_result_t train(trainer_data_t& data,
                const zob::stoch_optimizer optimizer, const size_t epochs, const bool verbose)
        {
                trainer_result_t result;

                const zob::timer_t timer;

                const auto batch_size = 16 * zob::n_threads();

                // set the sampling size
                data.m_tsampler.push(batch_size);

                // construct the optimization problem
                size_t epoch = 0;
                const size_t epoch_size = data.epoch_size(batch_size);

                auto fn_size = zob::make_opsize(data);
                auto fn_fval = zob::make_opfval(data);
                auto fn_grad = zob::make_opgrad(data);

                auto fn_ulog = [&] (opt_state_t& state, const auto& config)
                {
                        // evaluate training samples
                        data.m_lacc.set_params(state.x);
                        data.m_tsampler.pop();                  // revert to the original sampler
                        data.m_lacc.update(data.m_task, data.m_tsampler.get(), data.m_loss);
                        data.m_tsampler.push(batch_size);       // use the current minibatch sampler
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
                                zob::append(config, "lambda", data.lambda()));

                        if (verbose)
                        log_info()
                                << "[train = " << tvalue << "/" << terror_avg
                                << ", valid = " << vvalue << "/" << verror_avg
                                << " (" << zob::to_string(ret) << ")"
                                << ", epoch = " << epoch << "/" << epochs
                                << ", batch = " << batch_size
                                << ", " << append(config, "lambda", data.lambda())
                                << "] done in " << timer.elapsed() << ".";

                        state.f = tvalue;
                        return !zob::is_done(ret);
                };

                // Optimize the model
                zob::minimize(opt_problem_t(fn_size, fn_fval, fn_grad), fn_ulog,
                              data.m_x0, optimizer, epochs, epoch_size);

                // revert to the original sampler
                data.m_tsampler.pop();

                return result;
        }

        trainer_result_t stochastic_train(
                const model_t& model, const task_t& task, const fold_t& fold, const size_t nthreads,
                const loss_t& loss, const criterion_t& criterion,
                zob::stoch_optimizer optimizer, size_t epochs, bool verbose)
        {
                vector_t x0;
                model.save_params(x0);

                // setup accumulators
                accumulator_t lacc(model, criterion, criterion_t::type::value); lacc.set_threads(nthreads);
                accumulator_t gacc(model, criterion, criterion_t::type::vgrad); gacc.set_threads(nthreads);

                trainer_data_t data(task, fold, loss, x0, lacc, gacc);

                // tune the regularization factor (if needed)
                const auto op = [&] (scalar_t lambda)
                {
                        data.set_lambda(lambda);
                        return train(data, optimizer, epochs, verbose);
                };

                if (data.m_lacc.can_regularize())
                {
                        const auto space = zob::make_log10_space(-6.0, +6.0, 0.5);
                        return zob::tune(op, space).optimum();
                }
                else
                {
                        return op(0.0);
                }
        }
}
