#include "stochastic.h"
#include "../accumulator.h"
#include "../sampler.h"
#include "../util/logger.h"
#include "../util/log_search.hpp"
#include "../util/timer.h"
#include "../thread/pool.h"
#include "../minimize.h"

namespace ncv
{
        namespace detail
        {
                static scalar_t tune(
                        trainer_data_t& data,
                        stochastic_optimizer optimizer, size_t batch, scalar_t alpha0, scalar_t decay)
                {
                        data.setup(batch);

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
                        const opt_state_t state = ncv::minimize(
                                fn_size, fn_fval, fn_grad, fn_wlog, fn_elog, fn_ulog,
                                data.m_x0, optimizer, epochs, epoch_size, alpha0, decay);

                        // OK, cumulate the loss value
                        data.m_lacc.reset(state.x);
                        data.m_lacc.update(data.m_task, data.m_tsampler.all(), data.m_loss);
                        return data.m_lacc.value();
                }

                static trainer_result_t train(
                        trainer_data_t& data,
                        stochastic_optimizer optimizer, size_t batch, size_t epochs, scalar_t alpha0, scalar_t decay,
                        thread_pool_t::mutex_t& mutex, bool verbose)
                {
                        trainer_result_t result;

                        const ncv::timer_t timer;

                        data.setup(batch);

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
                                data.m_lacc.reset(state.x);
                                data.m_lacc.update(data.m_task, data.m_tsampler.all(), data.m_loss);
                                const scalar_t tvalue = data.m_lacc.value();
                                const scalar_t terror_avg = data.m_lacc.avg_error();
                                const scalar_t terror_var = data.m_lacc.var_error();

                                // evaluate validation samples
                                data.m_lacc.reset(state.x);
                                data.m_lacc.update(data.m_task, data.m_vsampler.all(), data.m_loss);
                                const scalar_t vvalue = data.m_lacc.value();
                                const scalar_t verror_avg = data.m_lacc.avg_error();
                                const scalar_t verror_var = data.m_lacc.var_error();

                                epoch ++;

                                // OK, update the optimum solution
                                const thread_pool_t::lock_t lock(mutex);

                                result.update(state.x, tvalue, terror_avg, terror_var, vvalue, verror_avg, verror_var,
                                              epoch, scalars_t({ alpha0, decay, data.m_lacc.lambda() }));

                                if (verbose)
                                log_info()
                                        << "[train = " << tvalue << "/" << terror_avg
                                        << ", valid = " << vvalue << "/" << verror_avg
                                        << ", xnorm = " << state.x.lpNorm<Eigen::Infinity>()
                                        << ", epoch = " << epoch << "/" << epochs
                                        << ", batch = " << batch
                                        << ", alpha = " << alpha0
                                        << ", decay = " << decay
                                        << ", lambda = " << data.m_lacc.lambda()
                                        << "] done in " << timer.elapsed() << ".";
                        };

                        // optimize the model
                        ncv::minimize(fn_size, fn_fval, fn_grad, fn_wlog, fn_elog, fn_ulog,
                                      data.m_x0, optimizer, epochs, epoch_size, alpha0, decay);

                        // OK
                        return result;
                }
        }

        trainer_result_t stochastic_train(
                const model_t& model, const task_t& task, const sampler_t& tsampler, const sampler_t& vsampler, size_t nthreads,
                const loss_t& loss, const string_t& criterion,
                stochastic_optimizer optimizer, size_t epochs,
                bool verbose)
        {
                thread_pool_t::mutex_t mutex;

                vector_t x0;
                model.save_params(x0);

                // operator to train for a given regularization factor
                const auto op = [&] (scalar_t lambda)
                {
                        scalar_t opt_state = std::numeric_limits<scalar_t>::max();
                        scalar_t opt_alpha = 1.00;
                        scalar_t opt_decay = 0.50;
                        size_t opt_batch = 0;

                        // operator to tune the learning rate
                        const auto op_lrate = [&] (scalar_t alpha)
                        {
                                accumulator_t lacc(model, nthreads, criterion, criterion_t::type::value, lambda);
                                accumulator_t gacc(model, nthreads, criterion, criterion_t::type::vgrad, lambda);

                                trainer_data_t data(task, tsampler, vsampler, loss, x0, lacc, gacc);

                                // tune the decay rate (if possible)
                                scalars_t decays;
                                switch (optimizer)
                                {
                                case stochastic_optimizer::AG:
                                case stochastic_optimizer::ADAGRAD:
                                case stochastic_optimizer::ADADELTA:
                                        decays = { 1.00 };
                                        break;

                                default:
                                        decays = { 0.0, 0.10, 0.20, 0.50, 0.75, 1.00 };
                                        break;
                                }

                                std::set<std::tuple<scalar_t, scalar_t, size_t>> states;
                                for (scalar_t decay : decays)
                                {                                        
                                        // tune the batch size
                                        const size_t min_batch = 16 * ncv::n_threads();
                                        const size_t max_batch = 16 * min_batch;

                                        for (size_t batch = min_batch; batch <= max_batch; batch *= 2)
                                        {
                                                const ncv::timer_t timer;

                                                const scalar_t state = detail::tune(data, optimizer, batch, alpha, decay);

                                                const thread_pool_t::lock_t lock(mutex);

                                                if (verbose)
                                                log_info()
                                                        << "[tuning: loss = " << state
                                                        << ", batch = " << batch
                                                        << ", alpha = " << alpha
                                                        << ", decay = " << decay
                                                        << ", lambda = " << data.m_lacc.lambda()
                                                        << "] done in " << timer.elapsed() << ".";

                                                states.insert(std::make_tuple(state, decay, batch));
                                        }
                                }

                                if (std::get<0>(*states.begin()) < opt_state)
                                {
                                        opt_alpha = alpha;
                                        opt_state = std::get<0>(*states.begin());
                                        opt_decay = std::get<1>(*states.begin());
                                        opt_batch = std::get<2>(*states.begin());
                                }

                                return std::get<0>(*states.begin());
                        };

                        // tune the learning rate (if possible)
                        switch (optimizer)
                        {
                        case stochastic_optimizer::ADADELTA:
                                op_lrate(1.0);
                                break;

                        default:
                                log10_min_search(op_lrate, -4.0, +2.0, 0.2, 4);
                                break;
                        }

                        // train the model using the tuned parameters
                        {
                                accumulator_t lacc(model, nthreads, criterion, criterion_t::type::value, lambda);
                                accumulator_t gacc(model, nthreads, criterion, criterion_t::type::vgrad, lambda);

                                trainer_data_t data(task, tsampler, vsampler, loss, x0, lacc, gacc);

                                return detail::train(data, optimizer, opt_batch, epochs, opt_alpha, opt_decay, mutex, verbose);
                        }
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
