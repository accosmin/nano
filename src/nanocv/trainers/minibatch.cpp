#include "batch.h"
#include "accumulator.h"
#include "sampler.h"
#include "file/logger.h"
#include "util/log_search.hpp"
#include "util/thread_pool.h"
#include "util/timer.h"
#include "optimize.h"

namespace ncv
{
        namespace detail
        {
                ///
                /// \brief restore the original samplers at destruction
                ///
                struct sampler_backup_t
                {
                        sampler_backup_t(trainer_data_t& data, size_t tsize)
                                :       m_data(data),
                                        m_tsampler_orig(data.m_tsampler),
                                        m_vsampler_orig(data.m_vsampler)
                        {
                                // random subset of training samples
                                if (tsize < m_tsampler_orig.size())
                                {
                                        data.m_tsampler.setup(sampler_t::stype::uniform, tsize);
                                }
                                else
                                {
                                        data.m_tsampler.setup(sampler_t::stype::batch);
                                }
                                data.m_tsampler = sampler_t(data.m_tsampler.get());

                                // all validation samples
                                data.m_vsampler.setup(sampler_t::stype::batch);
                        }

                        ~sampler_backup_t()
                        {
                                m_data.m_tsampler = m_tsampler_orig;
                                m_data.m_vsampler = m_vsampler_orig;
                        }

                        trainer_data_t&         m_data;
                        const sampler_t         m_tsampler_orig;
                        const sampler_t         m_vsampler_orig;
                };

                static scalar_t tune(
                        trainer_data_t& data,
                        batch_optimizer optimizer,
                        size_t batch, scalar_t ratio, size_t iterations, scalar_t epsilon)
                {
                        const size_t epochs = 8;

                        // construct the optimization problem
                        auto fn_size = ncv::make_opsize(data);
                        auto fn_fval = ncv::make_opfval(data);
                        auto fn_grad = ncv::make_opgrad(data);

                        auto fn_wlog = nullptr;
                        auto fn_elog = nullptr;
                        auto fn_ulog = nullptr;

                        // optimize the model
                        vector_t x = data.m_x0;

                        for (size_t epoch = 1, tsize = batch; epoch <= epochs;
                                epoch ++, tsize = static_cast<size_t>(tsize * ratio))
                        {
                                const sampler_backup_t sampler_data(data, tsize);

                                const opt_state_t state = ncv::minimize(
                                        fn_size, fn_fval, fn_grad, fn_wlog, fn_elog, fn_ulog,
                                        x, optimizer, iterations, epsilon);

                                x = state.x;
                        }

                        // OK, cumulate the loss value
                        data.m_lacc.reset(x);
                        data.m_lacc.update(data.m_task, data.m_tsampler.all(), data.m_loss);
                        return data.m_lacc.value();
                }

                static trainer_result_t train(
                        trainer_data_t& data,
                        batch_optimizer optimizer,
                        size_t epochs, size_t batch, scalar_t ratio, size_t iterations, scalar_t epsilon)
                {
                        trainer_result_t result;

                        const ncv::timer_t timer;

                        // construct the optimization problem
                        size_t iteration = 0, epoch = 1, tsize = batch;
                        const size_t vsize = data.m_vsampler.size();

                        auto fn_size = ncv::make_opsize(data);
                        auto fn_fval = ncv::make_opfval(data);
                        auto fn_grad = ncv::make_opgrad(data);

                        auto fn_wlog = ncv::make_opwlog();
                        auto fn_elog = ncv::make_opelog();
                        auto fn_ulog = [&] (const opt_state_t& state)
                        {
                                if (((++ iteration) % iterations) == 0)
                                {
                                        const scalar_t tvalue = data.m_gacc.value();
                                        const scalar_t terror_avg = data.m_gacc.avg_error();
                                        const scalar_t terror_var = data.m_gacc.var_error();

                                        // validation samples: loss value
                                        data.m_lacc.reset(state.x);
                                        data.m_lacc.update(data.m_task, data.m_vsampler.get(), data.m_loss);
                                        const scalar_t vvalue = data.m_lacc.value();
                                        const scalar_t verror_avg = data.m_lacc.avg_error();
                                        const scalar_t verror_var = data.m_lacc.var_error();

                                        // update the optimum state
                                        result.update(state.x, tvalue, terror_avg, terror_var, vvalue, verror_avg, verror_var,
                                                      epoch, scalars_t({ static_cast<scalar_t>(batch),
                                                                         ratio,
                                                                         static_cast<scalar_t>(iterations),
                                                                         data.m_lacc.lambda() }));

                                        log_info()
                                                << "[train = " << tvalue << "/" << terror_avg << "/=" << tsize
                                                << ", valid = " << vvalue << "/" << verror_avg << "/=" << vsize
                                                << ", xnorm = " << state.x.lpNorm<Eigen::Infinity>()
                                                << ", gnorm = " << state.g.lpNorm<Eigen::Infinity>()
                                                << ", epoch = " << epoch
                                                << ", batch = " << batch
                                                << ", ratio = " << ratio
                                                << ", iters = " << iterations
                                                << ", lambda = " << data.m_lacc.lambda()
                                                << ", calls = " << state.n_fval_calls() << "/" << state.n_grad_calls()
                                                << "] done in " << timer.elapsed() << ".";
                                }
                        };

                        // optimize the model
                        vector_t x = data.m_x0;

                        for (   tsize = batch; epoch <= epochs;
                                epoch ++, tsize = static_cast<size_t>(tsize * ratio))
                        {
                                const sampler_backup_t sampler_data(data, tsize);

                                const opt_state_t state = ncv::minimize(
                                        fn_size, fn_fval, fn_grad, fn_wlog, fn_elog, fn_ulog,
                                        x, optimizer, iterations, epsilon);

                                x = state.x;
                        }

                        return result;
                }
        }

        trainer_result_t minibatch_train(
                const model_t& model, const task_t& task, const sampler_t& tsampler, const sampler_t& vsampler, size_t nthreads,
                const loss_t& loss, const string_t& criterion,
                batch_optimizer optimizer, size_t epochs, scalar_t epsilon)
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

                        const scalars_t batch_ratios = { 1.00, 1.01, 1.02, 1.03 };
                        const indices_t batch_iterations = { 1, 2, 4, 8 };

                        scalar_t opt_state = std::numeric_limits<scalar_t>::max();
                        scalar_t opt_ratio = 1.0;
                        size_t opt_batch = min_batch;
                        size_t opt_iterations = 4;

                        // tune the batch size, the batch size ratio and the number of optimization iterations per batch
                        for (size_t batch = min_batch; batch <= max_batch; batch *= 2)
                        {
                                for (scalar_t ratio : batch_ratios)
                                {
                                        for (size_t iterations : batch_iterations)
                                        {
                                                const ncv::timer_t timer;

                                                const scalar_t state =
                                                        detail::tune(data, optimizer, batch, ratio, iterations, epsilon);

                                                log_info()
                                                        << "[tuning: loss = " << state
                                                        << ", batch = " << batch
                                                        << ", ratio = " << ratio
                                                        << ", iters = " << iterations
                                                        << ", lambda = " << data.m_lacc.lambda()
                                                        << "] done in " << timer.elapsed() << ".";

                                                if (state < opt_state)
                                                {
                                                        opt_state = state;
                                                        opt_batch = batch;
                                                        opt_ratio = ratio;
                                                        opt_iterations = iterations;
                                                }
                                        }
                                }
                        }

                        // train the model using the tuned learning rate & decay rate
                        return detail::train(data, optimizer, epochs, opt_batch, opt_ratio, opt_iterations, epsilon);
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
	
