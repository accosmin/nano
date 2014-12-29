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
                static trainer_result_t stochastic_train(
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

                        auto fn_size = [&] ()
                        {
                                return data.m_gacc.psize();
                        };

                        auto fn_fval = [&] (const vector_t& x)
                        {
                                data.m_lacc.reset(x);
                                data.m_lacc.update(data.m_task, tsamples[(index ++) % tsamples.size()], data.m_loss);

                                return data.m_lacc.value();
                        };

                        auto fn_fval_grad = [&] (const vector_t& x, vector_t& gx)
                        {
                                data.m_gacc.reset(x);
                                data.m_gacc.update(data.m_task, tsamples[(index ++) % tsamples.size()], data.m_loss);

                                gx = data.m_gacc.vgrad();
                                return data.m_gacc.value();
                        };

                        auto fn_wlog = [] (const string_t& message)
                        {
                                log_warning() << message;
                        };
                        auto fn_elog = [] (const string_t& message)
                        {
                                log_error() << message;
                        };
                        const opt_opulog_t fn_ulog = [&] (const opt_state_t& state)
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
                        ncv::minimize(fn_size, fn_fval, fn_fval_grad, fn_wlog, fn_elog, fn_ulog,
                                      data.m_x0, optimizer, epochs, tsamples.size(), alpha0);

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

                        // operator to tune the learning rate
                        const auto op_lrate = [&] (scalar_t alpha)
                        {
                                accumulator_t lacc(model, 1, criterion, criterion_t::type::value, lambda);
                                accumulator_t gacc(model, 1, criterion, criterion_t::type::vgrad, lambda);

                                trainer_data_t data(task, tsampler, vsampler, loss, x0, lacc, gacc);

                                // tune the decay rate
                                trainer_result_t result;
                                for (scalar_t decay : { 0.50, 0.75, 1.00 })
                                {
                                        result.update(detail::stochastic_train(data, optimizer, epochs, alpha, decay, mutex));
                                }

                                return result;
                        };

                        thread_pool_t wpool(nthreads);
                        return log_min_search_mt(op_lrate, wpool, -6.0, -1.0, 0.5, nthreads);
                };

                // tune the regularization factor (if needed)
                if (accumulator_t::can_regularize(criterion))
                {
                        return log_min_search(op, -2.0, +6.0, 0.5, 4);
                }

                else
                {
                        return op(0.0);
                }
        }
}
