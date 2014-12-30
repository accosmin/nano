#include "stochastic.h"
#include "accumulator.h"
#include "sampler.h"
#include "file/logger.h"
#include "util/log_search.hpp"
#include "util/random.hpp"
#include "util/thread_pool.h"
#include "util/timer.h"
#include "optimize.h"

namespace ncv
{
        namespace detail
        {
                static opt_state_t tune(
                        trainer_data_t& data,
                        stochastic_optimizer optimizer, scalar_t alpha0, scalar_t decay)
                {
                        const samples_t tsamples = data.m_tsampler.get();
                        const size_t epochs = 1;

                        // construct the optimization problem (NB: one random sample at the time)
                        size_t index = 0;

                        auto fn_size = ncv::make_opsize(data);
                        auto fn_fval = ncv::make_opfval(data, tsamples, index);
                        auto fn_grad = ncv::make_opgrad(data, tsamples, index);

                        auto fn_wlog = nullptr;
                        auto fn_elog = nullptr;
                        auto fn_ulog = nullptr;

                        // assembly optimization problem & optimize the model
                        opt_state_t opt = ncv::minimize(
                                fn_size, fn_fval, fn_grad, fn_wlog, fn_elog, fn_ulog,
                                data.m_x0, optimizer, epochs, tsamples.size(), alpha0, decay);

                        // OK, cumulate the loss value
                        data.m_lacc.reset(opt.x);
                        data.m_lacc.update(data.m_task, tsamples, data.m_loss);

                        opt.f = data.m_lacc.value();
                        return opt;
                }

                static trainer_result_t train(
                        trainer_data_t& data,
                        stochastic_optimizer optimizer, size_t epochs, scalar_t alpha0, scalar_t decay,
                        thread_pool_t::mutex_t& mutex)
                {
                        samples_t tsamples = data.m_tsampler.get();
                        samples_t vsamples = data.m_vsampler.get();

                        trainer_result_t result;

                        const ncv::timer_t timer;

                        // construct the optimization problem (NB: one random sample at the time)
                        size_t index = 0, epoch = 0;

                        auto fn_size = ncv::make_opsize(data);
                        auto fn_fval = ncv::make_opfval(data, tsamples, index);
                        auto fn_grad = ncv::make_opgrad(data, tsamples, index);

                        auto fn_wlog = ncv::make_opwlog();
                        auto fn_elog = ncv::make_opelog();
                        auto fn_ulog = [&] (const opt_state_t& state)
                        {
                                // shuffle randomly the training samples after each epoch
                                random_t<size_t> xrng(0, tsamples.size());
                                random_index_t<size_t> xrnd(xrng);

                                std::random_shuffle(tsamples.begin(), tsamples.end(), xrnd);

                                // evaluate training samples
                                data.m_lacc.reset(state.x);
                                data.m_lacc.update(data.m_task, tsamples, data.m_loss);
                                const scalar_t tvalue = data.m_lacc.value();
                                const scalar_t terror_avg = data.m_lacc.avg_error();
                                const scalar_t terror_var = data.m_lacc.var_error();

                                // evaluate validation samples
                                data.m_lacc.reset(state.x);
                                data.m_lacc.update(data.m_task, vsamples, data.m_loss);
                                const scalar_t vvalue = data.m_lacc.value();
                                const scalar_t verror_avg = data.m_lacc.avg_error();
                                const scalar_t verror_var = data.m_lacc.var_error();

                                epoch ++;

                                // OK, update the optimum solution
                                const thread_pool_t::lock_t lock(mutex);

                                result.update(state.x, tvalue, terror_avg, terror_var, vvalue, verror_avg, verror_var,
                                              epoch, scalars_t({ alpha0, decay, data.m_lacc.lambda() }));

                                log_info()
                                        << "[train = " << tvalue << "/" << terror_avg
                                        << ", valid = " << vvalue << "/" << verror_avg
                                        << ", xnorm = " << state.x.lpNorm<Eigen::Infinity>()
                                        << ", alpha = " << alpha0
                                        << ", decay = " << decay
                                        << ", epoch = " << epoch << "/" << epochs
                                        << ", lambda = " << data.m_lacc.lambda()
                                        << "] done in " << timer.elapsed() << ".";
                        };

                        // assembly optimization problem & optimize the model
                        ncv::minimize(fn_size, fn_fval, fn_grad, fn_wlog, fn_elog, fn_ulog,
                                      data.m_x0, optimizer, epochs, tsamples.size(), alpha0, decay);

                        // OK
                        return result;
                }
        }

        trainer_result_t stochastic_train(
                const model_t& model, const task_t& task, const sampler_t& tsampler, const sampler_t& vsampler, size_t nthreads,
                const loss_t& loss, const string_t& criterion,
                stochastic_optimizer optimizer, size_t epochs)
        {
                thread_pool_t::mutex_t mutex;

                // operator to train for a given regularization factor
                const auto op = [&] (scalar_t lambda)
                {
                        vector_t x0;
                        model.save_params(x0);                        

                        opt_state_t opt_state;
                        scalar_t opt_alpha = 1.00;
                        scalar_t opt_decay = 0.50;

                        // operator to tune the learning rate
                        const auto op_lrate = [&] (scalar_t alpha)
                        {
                                accumulator_t lacc(model, 1, criterion, criterion_t::type::value, lambda);
                                accumulator_t gacc(model, 1, criterion, criterion_t::type::vgrad, lambda);

                                trainer_data_t data(task, tsampler, vsampler, loss, x0, lacc, gacc);

                                const scalars_t decays = { 0.50, 0.75, 1.00 };

                                // also tune the decay rate
                                std::set<std::pair<opt_state_t, scalar_t> > states;
                                for (scalar_t decay : decays)
                                {
                                        const ncv::timer_t timer;

                                        const opt_state_t state = detail::tune(data, optimizer, alpha, decay);

                                        const thread_pool_t::lock_t lock(mutex);

                                        log_info()
                                                << "[tuning: loss = " << state.f
                                                << ", alpha = " << alpha
                                                << ", decay = " << decay
                                                << ", lambda = " << data.m_lacc.lambda()
                                                << "] done in " << timer.elapsed() << ".";

                                        states.insert(std::make_pair(state, decay));
                                }

                                if (states.begin()->first < opt_state)
                                {
                                        opt_state = states.begin()->first;
                                        opt_alpha = alpha;
                                        opt_decay = states.begin()->second;
                                }

                                return states.begin()->first;
                        };

                        thread_pool_t wpool(nthreads);
                        log_min_search_mt(op_lrate, wpool, -6.0, 0.0, 0.5, nthreads);

                        // train the model using the tuned learning rate & decay rate
                        {
                                accumulator_t lacc(model, 1, criterion, criterion_t::type::value, lambda);
                                accumulator_t gacc(model, 1, criterion, criterion_t::type::vgrad, lambda);

                                trainer_data_t data(task, tsampler, vsampler, loss, x0, lacc, gacc);

                                return detail::train(data, optimizer, epochs, opt_alpha, opt_decay, mutex);
                        }
                };

                // tune the regularization factor (if needed)
                if (accumulator_t::can_regularize(criterion))
                {
                        thread_pool_t wpool(nthreads);
                        return log_min_search_mt(op, wpool, -4.0, +4.0, 0.5, nthreads).first;
                }

                else
                {
                        return op(0.0);
                }
        }
}
