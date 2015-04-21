#include "minibatch.h"
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
                void setup_minibatch(sampler_t& orig_tsampler, size_t tsize, trainer_data_t& data)
                {
                        // FIXED random subset of training samples
                        orig_tsampler.setup(sampler_t::stype::uniform, tsize);
                        data.m_tsampler = sampler_t(orig_tsampler.get());
                }

                void reset_minibatch(sampler_t& orig_tsampler, trainer_data_t& data)
                {
                        // all available training samples
                        orig_tsampler.setup(sampler_t::stype::batch);
                        data.m_tsampler = orig_tsampler;
                }

                size_t make_epoch_size(const trainer_data_t& data, size_t batch)
                {
                        return (data.m_tsampler.size() + batch - 1) / batch;
                }

                template
                <
                        typename toperator
                >
                void train(trainer_data_t& data, size_t epoch_size, size_t batch, const toperator& op)
                {
                        sampler_t orig_tsampler = data.m_tsampler;

                        for (size_t i = 0; i < epoch_size; i ++)
                        {
                                setup_minibatch(orig_tsampler, batch, data);
                                op();
                        }

                        reset_minibatch(orig_tsampler, data);
                }

                trainer_result_t train(
                        trainer_data_t& data,
                        optim::batch_optimizer optimizer,
                        size_t epochs, size_t batch, size_t iterations, scalar_t epsilon,
                        bool verbose)
                {
                        const ncv::timer_t timer;

                        trainer_result_t result;

                        const size_t epoch_size = make_epoch_size(data, batch);

                        // construct the optimization problem
                        auto fn_size = ncv::make_opsize(data);
                        auto fn_fval = ncv::make_opfval(data);
                        auto fn_grad = ncv::make_opgrad(data);

                        auto fn_wlog = verbose ? ncv::make_opwlog() : nullptr;
                        auto fn_elog = verbose ? ncv::make_opelog() : nullptr;
                        auto fn_ulog = nullptr;

                        // optimize the model
                        vector_t x = data.m_x0;

                        for (size_t epoch = 1; epoch <= epochs; epoch ++)
                        {
                                train(data, epoch_size, batch, [&] ()
                                {
                                        const opt_state_t state = ncv::minimize(
                                                fn_size, fn_fval, fn_grad, fn_wlog, fn_elog, fn_ulog,
                                                x, optimizer, iterations, epsilon);

                                        x = state.x;
                                });

                                // training samples: loss value
                                data.m_lacc.set_params(x);
                                data.m_lacc.update(data.m_task, data.m_tsampler.all(), data.m_loss);
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
                                const auto ret = result.update(
                                        x, tvalue, terror_avg, terror_var, vvalue, verror_avg, verror_var,
                                        epoch, scalars_t({ static_cast<scalar_t>(batch),
                                                           static_cast<scalar_t>(iterations),
                                                           data.lambda() }));

                                if (verbose)
                                log_info()
                                        << "[train = " << tvalue << "/" << terror_avg
                                        << ", valid = " << vvalue << "/" << verror_avg
                                        << " (" << text::to_string(ret) << ")"
                                        << ", xnorm = " << x.lpNorm<Eigen::Infinity>()
                                        << ", epoch = " << epoch << "/" << epochs
                                        << ", batch = " << batch
                                        << ", iters = " << iterations
                                        << ", lambda = " << data.lambda()
                                        << "] done in " << timer.elapsed() << ".";

                                if (ret == trainer_result_return_t::overfitting)
                                {
                                        break;
                                }
                        }

                        return result;
                }

                // <result, batch size, iterations per batch>
                std::tuple<trainer_result_t, size_t, size_t> tune_minibatch(
                        trainer_data_t& data, optim::batch_optimizer optimizer, scalar_t epsilon,
                        bool verbose)
                {
                        const size_t min_batch = 16 * ncv::n_threads();
                        const size_t max_batch = 16 * min_batch;

                        const indices_t batch_iterations = { 4, 8 };

                        trainer_result_t opt_result;
                        size_t opt_batch = min_batch;
                        size_t opt_iterations = 4;

                        // tune the batch size and the number of optimization iterations per batch
                        for (size_t batch = min_batch; batch <= max_batch; batch *= 2)
                        {
                                for (size_t iterations : batch_iterations)
                                {
                                        const ncv::timer_t timer;

                                        const size_t epochs = 1;
                                        const trainer_result_t result =
                                                train(data, optimizer, epochs, batch, iterations, epsilon, false);

                                        const trainer_state_t state = result.optimum_state();

                                        if (verbose)
                                        log_info()
                                                << "[tuning: train = " << state.m_tvalue << "/" << state.m_terror_avg
                                                << ", valid = " << state.m_vvalue << "/" << state.m_verror_avg
                                                << ", batch = " << batch
                                                << ", iters = " << iterations
                                                << ", lambda = " << data.lambda()
                                                << "] done in " << timer.elapsed() << ".";

                                        if (result < opt_result)
                                        {
                                                opt_result = result;
                                                opt_batch = batch;
                                                opt_iterations = iterations;
                                        }
                                }
                        }

                        // OK
                        return std::make_tuple(opt_result, opt_batch, opt_iterations);
                }
        }

        trainer_result_t minibatch_train(
                const model_t& model,
                const task_t& task, const sampler_t& tsampler, const sampler_t& vsampler, size_t nthreads,
                const loss_t& loss, const string_t& criterion,
                optim::batch_optimizer optimizer, size_t epochs, scalar_t epsilon, bool verbose)
        {
                vector_t x0;
                model.save_params(x0);

                // setup acumulators
                accumulator_t lacc(model, nthreads, criterion, criterion_t::type::value);
                accumulator_t gacc(model, nthreads, criterion, criterion_t::type::vgrad);

                trainer_data_t data(task, tsampler, vsampler, loss, x0, lacc, gacc);

                // tune the regularization factor (if needed)
                const auto op = [&] (scalar_t lambda)
                {
                        data.set_lambda(lambda);

                        const auto ret = tune_minibatch(data, optimizer, epsilon, verbose);

                        const size_t opt_batch = std::get<1>(ret);
                        const size_t opt_iterations = std::get<2>(ret);

                        return train(data, optimizer, epochs, opt_batch, opt_iterations, epsilon, verbose);
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
	
