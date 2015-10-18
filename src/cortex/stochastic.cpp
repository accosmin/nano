#include "sampler.h"
#include "timer.h"
#include "logger.h"
#include "minimize.h"
#include "stochastic.h"
#include "accumulator.h"
#include "thread/thread.h"
#include "text/to_string.hpp"
#include "math/tune_fixed.hpp"
#include "math/tune_log10.hpp"
#include <tuple>

namespace cortex
{
        namespace
        {
                sizes_t tunable_batches()
                {
                        const size_t batch0 = thread::n_threads();

                        return { batch0, batch0 * 2, batch0 * 4, batch0 * 8, batch0 * 16 };
                }

                trainer_result_t train(
                        trainer_data_t& data,
                        min::stoch_optimizer optimizer, size_t epochs, size_t batch, scalar_t alpha0, scalar_t decay,
                        bool verbose)
                {
                        trainer_result_t result;

                        const cortex::timer_t timer;

                        data.set_batch(batch);

                        // construct the optimization problem
                        size_t epoch = 0;
                        const size_t epoch_size = (data.m_tsampler.size() + batch - 1) / batch;

                        auto fn_size = cortex::make_opsize(data);
                        auto fn_fval = cortex::make_opfval(data);
                        auto fn_grad = cortex::make_opgrad(data);

                        auto fn_ulog = [&] (const opt_state_t& state)
                        {
                                // evaluate training samples
                                data.m_lacc.set_params(state.x);
                                data.m_lacc.update(data.m_task, data.m_tsampler.all(), data.m_loss);
                                const scalar_t tvalue = data.m_lacc.value();
                                const scalar_t terror_avg = data.m_lacc.avg_error();
                                const scalar_t terror_var = data.m_lacc.var_error();

                                // evaluate validation samples
                                data.m_lacc.set_params(state.x);
                                data.m_lacc.update(data.m_task, data.m_vsampler.all(), data.m_loss);
                                const scalar_t vvalue = data.m_lacc.value();
                                const scalar_t verror_avg = data.m_lacc.avg_error();
                                const scalar_t verror_var = data.m_lacc.var_error();

                                epoch ++;

                                // OK, update the optimum solution
                                const auto ret = result.update(
                                        state.x, tvalue, terror_avg, terror_var, vvalue, verror_avg, verror_var,
                                        epoch, scalars_t({ static_cast<scalar_t>(batch),
                                                           alpha0,
                                                           decay,
                                                           data.lambda() }));

                                if (verbose)
                                log_info()
                                        << "[train = " << tvalue << "/" << terror_avg
                                        << ", valid = " << vvalue << "/" << verror_avg
                                        << " (" << text::to_string(ret) << ")"
                                        << ", epoch = " << epoch << "/" << epochs
                                        << ", batch = " << batch
                                        << ", alpha = " << alpha0
                                        << ", decay = " << decay
                                        << ", lambda = " << data.lambda()
                                        << "] done in " << timer.elapsed() << ".";

                                return !cortex::is_done(ret);
                        };

                        // OK, optimize the model
                        min::minimize(opt_problem_t(fn_size, fn_fval, fn_grad), fn_ulog,
                                      data.m_x0, optimizer, epochs, epoch_size, alpha0, decay);

                        return result;
                }

                // <result, batch size, decay rate, learning rate>
                auto tune_batch_decay_lrate(trainer_data_t& data, min::stoch_optimizer optimizer, bool verbose)
                {
                        const auto op = [&] (size_t batch, scalar_t decay, scalar_t alpha)
                        {
                                const cortex::timer_t timer;

                                const size_t epochs = 1;
                                const auto result = train(data, optimizer, epochs, batch, alpha, decay, false);
                                const auto state = result.optimum_state();

                                if (verbose)
                                log_info()
                                        << "[tuning: train = " << state.m_tvalue << "/" << state.m_terror_avg
                                        << ", valid = " << state.m_vvalue << "/" << state.m_verror_avg
                                        << ", batch = " << batch
                                        << ", alpha = " << alpha
                                        << ", decay = " << decay
                                        << ", lambda = " << data.lambda()
                                        << "] done in " << timer.elapsed() << ".";

                                return result;
                        };

                        const auto batches = tunable_batches();
                        const auto decays = tunable_decays(optimizer);
                        const auto alphas = tunable_alphas(optimizer);

                        return math::tune_fixed(op, batches, decays, alphas);
                }
        }

        trainer_result_t stochastic_train(
                const model_t& model,
                const task_t& task, const sampler_t& tsampler, const sampler_t& vsampler, size_t nthreads,
                const loss_t& loss, const string_t& criterion,
                min::stoch_optimizer optimizer, size_t epochs, bool verbose)
        {
                vector_t x0;
                model.save_params(x0);

                // setup accumulators
                accumulator_t lacc(model, nthreads, criterion, criterion_t::type::value);
                accumulator_t gacc(model, nthreads, criterion, criterion_t::type::vgrad);

                trainer_data_t data(task, tsampler, vsampler, loss, x0, lacc, gacc);

                // tune the regularization factor (if needed)
                const auto op = [&] (scalar_t lambda)
                {
                        data.set_lambda(lambda);

                        const auto ret = tune_batch_decay_lrate(data, optimizer, verbose);
                        const auto opt_batch = std::get<1>(ret);
                        const auto opt_decay = std::get<2>(ret);
                        const auto opt_alpha = std::get<3>(ret);

                        return train(data, optimizer, epochs, opt_batch, opt_alpha, opt_decay, verbose);
                };

                if (data.m_lacc.can_regularize())
                {
                        return std::get<0>(math::tune_log10(op, -6.0, +0.0, 0.5, 4));
                }
                else
                {
                        return op(0.0);
                }
        }
}
