#include "stochastic.h"
#include "nanocv/timer.h"
#include "nanocv/logger.h"
#include "nanocv/sampler.h"
#include "nanocv/minimize.h"
#include "nanocv/accumulator.h"
#include "nanocv/log_search.hpp"
#include "nanocv/thread/thread.h"
#include <tuple>

namespace ncv
{
        namespace
        {
                scalars_t tunable_alphas(const optim::stoch_optimizer optimizer)
                {
                        switch (optimizer)
                        {
                        case optim::stoch_optimizer::ADADELTA:
                                return { 0.0 };

                        default:
                                return { 1e-4, 1e-3, 1e-2, 1e-1 };
                        }
                }

                scalars_t tunable_decays(const optim::stoch_optimizer optimizer)
                {
                        switch (optimizer)
                        {
                        case optim::stoch_optimizer::AG:
                        case optim::stoch_optimizer::ADAGRAD:
                        case optim::stoch_optimizer::ADADELTA:
                                return { 1.00 };

                        default:
                                return { 0.10, 0.20, 0.50, 0.75, 1.00 };
                        }
                }

                trainer_result_t train(
                        trainer_data_t& data,
                        optim::stoch_optimizer optimizer, size_t epochs, size_t batch, scalar_t alpha0, scalar_t decay,
                        bool verbose)
                {
                        trainer_result_t result;

                        const ncv::timer_t timer;

                        data.set_batch(batch);

                        // construct the optimization problem
                        size_t epoch = 0;
                        const size_t epoch_size = (data.m_tsampler.size() + batch - 1) / batch;

                        auto fn_size = ncv::make_opsize(data);
                        auto fn_fval = ncv::make_opfval(data);
                        auto fn_grad = ncv::make_opgrad(data);

                        auto fn_wlog = verbose ? ncv::make_opwlog() : nullptr;
                        auto fn_elog = verbose ? ncv::make_opelog() : nullptr;
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

                                return !ncv::is_done(ret);
                        };

                        // OK, optimize the model
                        ncv::minimize(fn_size, fn_fval, fn_grad, fn_wlog, fn_elog, fn_ulog,
                                      data.m_x0, optimizer, epochs, epoch_size, alpha0, decay);

                        return result;
                }

                // <result, batch size, decay rate, learning rate>
                decltype(auto) tune_batch_decay_lrate(trainer_data_t& data,
                        optim::stoch_optimizer optimizer, bool verbose)
                {
                        trainer_result_t opt_result;
                        scalar_t opt_decay = 0.0;
                        scalar_t opt_alpha = 0.0;
                        size_t opt_batch = 0;

                        const scalars_t decays = tunable_decays(optimizer);
                        const scalars_t alphas = tunable_alphas(optimizer);

                        // tune the decay rate
                        for (scalar_t decay : decays)
                        {
                                // tune the learning rate
                                for (scalar_t alpha : alphas)
                                {
                                        // tune the batch size
                                        const size_t min_batch = 16 * ncv::n_threads();
                                        const size_t max_batch = 16 * min_batch;

                                        for (size_t batch = min_batch; batch <= max_batch; batch *= 2)
                                        {
                                                const ncv::timer_t timer;

                                                const size_t epochs = 1;
                                                const auto result = train(data, optimizer, epochs, batch, alpha, decay, false);

                                                const trainer_state_t state = result.optimum_state();

                                                if (verbose)
                                                log_info()
                                                        << "[tuning: train = " << state.m_tvalue << "/" << state.m_terror_avg
                                                        << ", valid = " << state.m_vvalue << "/" << state.m_verror_avg
                                                        << ", batch = " << batch
                                                        << ", alpha = " << alpha
                                                        << ", decay = " << decay
                                                        << ", lambda = " << data.lambda()
                                                        << "] done in " << timer.elapsed() << ".";

                                                if (result < opt_result)
                                                {
                                                        opt_result = result;
                                                        opt_batch = batch;
                                                        opt_decay = decay;
                                                        opt_alpha = alpha;
                                                }
                                        }
                                }
                        }

                        // OK
                        return std::make_tuple(opt_result, opt_batch, opt_decay, opt_alpha);
                }

                // <result, batch size, decay rate, learning rate, regularization weight>
                decltype(auto) tune_lambda(trainer_data_t& data,
                        optim::stoch_optimizer optimizer, bool verbose)
                {
                        const auto op = [&] (scalar_t lambda)
                        {
                                data.set_lambda(lambda);

                                const auto ret = tune_batch_decay_lrate(data, optimizer, verbose);
                                return std::tuple_cat(ret, std::make_tuple(lambda));
                        };

                        if (data.m_lacc.can_regularize())
                        {
                                return log10_min_search(op, -6.0, +0.0, 0.5, 4).first;
                        }
                        else
                        {
                                return op(0.0);
                        }
                }
        }

        trainer_result_t stochastic_train(
                const model_t& model,
                const task_t& task, const sampler_t& tsampler, const sampler_t& vsampler, size_t nthreads,
                const loss_t& loss, const string_t& criterion,
                optim::stoch_optimizer optimizer, size_t epochs, bool verbose)
        {
                vector_t x0;
                model.save_params(x0);

                // setup accumulators
                accumulator_t lacc(model, nthreads, criterion, criterion_t::type::value);
                accumulator_t gacc(model, nthreads, criterion, criterion_t::type::vgrad);

                trainer_data_t data(task, tsampler, vsampler, loss, x0, lacc, gacc);

                // tune the regularization factor (if needed)
                const auto ret = tune_lambda(data, optimizer, verbose);

                const size_t opt_batch = std::get<1>(ret);
                const scalar_t opt_decay = std::get<2>(ret);
                const scalar_t opt_alpha = std::get<3>(ret);
                const scalar_t opt_lambda = std::get<4>(ret);

                data.set_lambda(opt_lambda);

                return train(data, optimizer, epochs, opt_batch, opt_alpha, opt_decay, verbose);
        }
}
