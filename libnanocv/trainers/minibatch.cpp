#include "batch.h"
#include "libnanocv/accumulator.h"
#include "libnanocv/sampler.h"
#include "libnanocv/util/logger.h"
#include "libnanocv/util/log_search.hpp"
#include "libnanocv/util/thread.h"
#include "libnanocv/util/timer.h"
#include "libnanocv/optimize.h"

namespace ncv
{
        namespace detail
        {
                ///
                /// \brief restore the original sampler at destruction
                ///
                struct sampler_backup_t
                {
                        sampler_backup_t(trainer_data_t& data, size_t tsize)
                                :       m_data(data),
                                        m_tsampler_orig(data.m_tsampler)
                        {
                                // FIXED random subset of training samples
                                data.m_tsampler.setup(sampler_t::stype::uniform, tsize);
                                data.m_tsampler = sampler_t(data.m_tsampler.get());
                        }

                        ~sampler_backup_t()
                        {
                                m_data.m_tsampler = m_tsampler_orig;
                        }

                        trainer_data_t&         m_data;
                        const sampler_t         m_tsampler_orig;
                };

                static scalar_t tune(
                        trainer_data_t& data,
                        batch_optimizer optimizer,
                        size_t batch, size_t iterations, scalar_t epsilon)
                {
                        const size_t epochs = 1;
                        const size_t epoch_size = (data.m_tsampler.size() + batch - 1) / batch;

                        // construct the optimization problem
                        auto fn_size = ncv::make_opsize(data);
                        auto fn_fval = ncv::make_opfval(data);
                        auto fn_grad = ncv::make_opgrad(data);

                        auto fn_wlog = nullptr;
                        auto fn_elog = nullptr;
                        auto fn_ulog = nullptr;

                        // optimize the model
                        vector_t x = data.m_x0;

                        for (size_t epoch = 1; epoch <= epochs; epoch ++)
                        {
                                for (size_t i = 0; i < epoch_size; i ++)
                                {
                                        const sampler_backup_t sampler_data(data, batch);

                                        const opt_state_t state = ncv::minimize(
                                                fn_size, fn_fval, fn_grad, fn_wlog, fn_elog, fn_ulog,
                                                x, optimizer, iterations, epsilon);

                                        x = state.x;
                                }
                        }

                        // OK, cumulate the loss value
                        data.m_lacc.reset(x);
                        data.m_lacc.update(data.m_task, data.m_tsampler.all(), data.m_loss);
                        return data.m_lacc.value();
                }

                static trainer_result_t train(
                        trainer_data_t& data,
                        batch_optimizer optimizer,
                        size_t epochs, size_t batch, size_t iterations, scalar_t epsilon, bool verbose)
                {
                        trainer_result_t result;

                        const ncv::timer_t timer;

                        const size_t epoch_size = (data.m_tsampler.size() + batch - 1) / batch;

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
                                for (size_t i = 0; i < epoch_size; i ++)
                                {
                                        const sampler_backup_t sampler_data(data, batch);

                                        const opt_state_t state = ncv::minimize(
                                                fn_size, fn_fval, fn_grad, fn_wlog, fn_elog, fn_ulog,
                                                x, optimizer, iterations, epsilon);

                                        x = state.x;
                                }

                                // training samples: loss value
                                data.m_lacc.reset(x);
                                data.m_lacc.update(data.m_task, data.m_tsampler.all(), data.m_loss);
                                const scalar_t tvalue = data.m_lacc.value();
                                const scalar_t terror_avg = data.m_lacc.avg_error();
                                const scalar_t terror_var = data.m_lacc.var_error();

                                // validation samples: loss value
                                data.m_lacc.reset(x);
                                data.m_lacc.update(data.m_task, data.m_vsampler.get(), data.m_loss);
                                const scalar_t vvalue = data.m_lacc.value();
                                const scalar_t verror_avg = data.m_lacc.avg_error();
                                const scalar_t verror_var = data.m_lacc.var_error();

                                // update the optimum state
                                result.update(x, tvalue, terror_avg, terror_var, vvalue, verror_avg, verror_var,
                                              epoch, scalars_t({ static_cast<scalar_t>(batch),
                                                                 static_cast<scalar_t>(iterations),
                                                                 data.m_lacc.lambda() }));

                                if (verbose)
                                log_info()
                                        << "[train = " << tvalue << "/" << terror_avg
                                        << ", valid = " << vvalue << "/" << verror_avg
                                        << ", xnorm = " << x.lpNorm<Eigen::Infinity>()
                                        << ", epoch = " << epoch << "/" << epochs
                                        << ", batch = " << batch
                                        << ", iters = " << iterations
                                        << ", lambda = " << data.m_lacc.lambda()
                                        << "] done in " << timer.elapsed() << ".";
                        }

                        return result;
                }
        }

        trainer_result_t minibatch_train(
                const model_t& model, const task_t& task, const sampler_t& tsampler, const sampler_t& vsampler, size_t nthreads,
                const loss_t& loss, const string_t& criterion,
                batch_optimizer optimizer, size_t epochs, scalar_t epsilon,
                bool verbose)
        {
                vector_t x0;
                model.save_params(x0);

                // operator to train for a given regularization factor
                const auto op = [&] (scalar_t lambda)
                {
                        accumulator_t lacc(model, nthreads, criterion, criterion_t::type::value, lambda);
                        accumulator_t gacc(model, nthreads, criterion, criterion_t::type::vgrad, lambda);

                        trainer_data_t data(task, tsampler, vsampler, loss, x0, lacc, gacc);

                        const size_t min_batch = 16 * ncv::n_threads();
                        const size_t max_batch = 16 * min_batch;

                        const indices_t batch_iterations = { 1, 2, 4, 8 };

                        scalar_t opt_state = std::numeric_limits<scalar_t>::max();
                        size_t opt_batch = min_batch;
                        size_t opt_iterations = 4;

                        // tune the batch size and the number of optimization iterations per batch
                        for (size_t batch = min_batch; batch <= max_batch; batch *= 2)
                        {
                                for (size_t iterations : batch_iterations)
                                {
                                        const ncv::timer_t timer;

                                        const scalar_t state =
                                                detail::tune(data, optimizer, batch, iterations, epsilon);

                                        if (verbose)
                                        log_info()
                                                << "[tuning: loss = " << state
                                                << ", batch = " << batch
                                                << ", iters = " << iterations
                                                << ", lambda = " << data.m_lacc.lambda()
                                                << "] done in " << timer.elapsed() << ".";

                                        if (state < opt_state)
                                        {
                                                opt_state = state;
                                                opt_batch = batch;
                                                opt_iterations = iterations;
                                        }
                                }
                        }

                        // train the model using the tuned parameters
                        return detail::train(data, optimizer, epochs, opt_batch, opt_iterations, epsilon, verbose);
                };

                // tune the regularization factor (if needed)
                if (accumulator_t::can_regularize(criterion))
                {
                        return log10_min_search(op, -4.0, +4.0, 0.2, 4).first;
                }

                else
                {
                        return op(0.0);
                }
        }
}
	
