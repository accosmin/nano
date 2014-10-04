#include "stochastic.h"
#include "accumulator.h"
#include "sampler.h"
#include "common/log_search.hpp"
#include "common/random.hpp"
#include "common/thread_pool.h"
#include "common/logger.h"
#include "common/timer.h"

namespace ncv
{
        namespace detail
        {
                struct rnd_t
                {
                        rnd_t(random_t<size_t>& gen)
                                :       m_gen(gen)
                        {
                        }

                        size_t operator()(size_t i)
                        {
                                return m_gen() % i;
                        }

                        random_t<size_t>&  m_gen;
                };

                static trainer_result_t stochastic_train(
                        trainer_data_t& data,
                        stochastic_optimizer type, size_t epochs, scalar_t alpha0, scalar_t beta,
                        thread_pool_t::mutex_t& mutex)
                {
                        samples_t tsamples = data.m_tsampler.get();
                        samples_t vsamples = data.m_vsampler.get();

                        trainer_result_t result;

                        random_t<size_t> xrng(0, tsamples.size());
                        rnd_t xrnd(xrng);

                        timer_t timer;

                        vector_t x = data.m_x0, xparam = x, xavg = x;

                        vector_t gavg(x.size());
                        gavg.setZero();

                        scalar_t alpha = alpha0;
                        scalar_t sumb = 1.0 / alpha;

                        for (size_t e = 0; e < epochs; e ++)
                        {
                                std::random_shuffle(tsamples.begin(), tsamples.end(), xrnd);

                                // one epoch: a pass through all training samples
                                switch (type)
                                {
                                        // stochastic gradient
                                case stochastic_optimizer::SG:
                                        for (size_t i = 0; i < tsamples.size(); i ++, alpha *= beta)
                                        {
                                                data.m_gacc.reset(x);
                                                data.m_gacc.update(data.m_task, tsamples[i], data.m_loss);

                                                x.noalias() -= alpha * data.m_gacc.vgrad();
                                        }
                                        xparam = x;
                                        break;

                                        // stochastic gradient average
                                case stochastic_optimizer::SGA:
                                        for (size_t i = 0; i < tsamples.size(); i ++, alpha *= beta)
                                        {
                                                data.m_gacc.reset(x);
                                                data.m_gacc.update(data.m_task, tsamples[i], data.m_loss);

                                                const vector_t g = data.m_gacc.vgrad();

                                                const scalar_t b = 1.0 / alpha;
                                                gavg = (gavg * sumb + g * b) / (sumb + b);
                                                sumb = sumb + b;

                                                x.noalias() -= alpha * gavg;
                                        }
                                        xparam = x;
                                        break;

                                        // stochastic iterative average
                                case stochastic_optimizer::SIA:
                                default:
                                        for (size_t i = 0; i < tsamples.size(); i ++, alpha *= beta)
                                        {
                                                data.m_gacc.reset(x);
                                                data.m_gacc.update(data.m_task, tsamples[i], data.m_loss);

                                                x.noalias() -= alpha * data.m_gacc.vgrad();

                                                const scalar_t b = 1.0 / alpha;
                                                xavg = (xavg * sumb + x * b) / (sumb + b);
                                                sumb = sumb + b;
                                        }
                                        xparam = xavg;
                                        break;
                                }

                                // evaluate training samples
                                data.m_lacc.reset(xparam);
                                data.m_lacc.update(data.m_task, tsamples, data.m_loss);
                                const scalar_t tvalue = data.m_lacc.value();
                                const scalar_t terror = data.m_lacc.error();

                                // evaluate validation samples
                                data.m_lacc.reset(xparam);
                                data.m_lacc.update(data.m_task, vsamples, data.m_loss);
                                const scalar_t vvalue = data.m_lacc.value();
                                const scalar_t verror = data.m_lacc.error();

                                // OK, update the optimum solution
                                const thread_pool_t::lock_t lock(mutex);

                                result.update(xparam, tvalue, terror, vvalue, verror, e,
                                              scalars_t({ alpha0, data.m_lacc.lambda() }));

                                log_info()
                                        << "[train = " << tvalue << "/" << terror
                                        << ", valid = " << vvalue << "/" << verror
                                        << ", rate = " << alpha << "/" << alpha0
                                        << ", epoch = " << e << "/" << epochs
                                        << ", dims = " << data.m_lacc.psize()
                                        << ", lambda = " << data.m_lacc.lambda()
                                        << "] done in " << timer.elapsed() << ".";
                        }

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
                        const size_t iterations = epochs * tsampler.size();             // SGD iterations
                        const scalar_t beta = std::pow(0.01, 1.0 / iterations);         // Learning rate decay rate

                        vector_t x0;
                        model.save_params(x0);

                        // operator to tune the learning rate
                        const auto op_lrate = [&] (scalar_t alpha)
                        {
                                accumulator_t lacc(model, 1, criterion, criterion_t::type::value, lambda);
                                accumulator_t gacc(model, 1, criterion, criterion_t::type::vgrad, lambda);

                                trainer_data_t data(task, tsampler, vsampler, loss, x0, lacc, gacc);

                                return detail::stochastic_train(data, optimizer, epochs, alpha, beta, mutex);
                        };

                        thread_pool_t wpool(nthreads);

                        return log_min_search_mt(op_lrate, wpool, -6.0, -1.0, 0.5, nthreads);
                };

                // tune the regularization factor (if needed)
                if (accumulator_t::can_regularize(criterion))
                {
                        thread_pool_t wpool(nthreads);

                        return log_min_search_mt(op, wpool, -2.0, +6.0, 0.2, nthreads);
                }

                else
                {
                        return op(0.0);
                }
        }
}
