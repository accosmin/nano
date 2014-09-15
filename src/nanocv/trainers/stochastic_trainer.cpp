#include "stochastic_trainer.h"
#include "common/timer.h"
#include "common/logger.h"
#include "common/math.hpp"
#include "common/thread_pool.h"
#include "common/random.hpp"
#include "common/log_search.hpp"
#include "sampler.h"
#include "accumulator.h"

namespace ncv
{
        namespace detail
        {
                ///
                /// \brief tune the regularization factor for a given criterion (if needed)
                ///
                template
                <
                        typename toperator
                >
                static trainer_result_t tune(const toperator& op, const string_t& criterion)
                {
                        if (accumulator_t::can_regularize(criterion))
                        {
                                return log_min_search<toperator, scalar_t>(op, -1.0, +6.0, 0.2);
                        }

                        else
                        {
                                return op(0.0);
                        }
                }

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

                static void stochastic_train(
                        trainer_data_t& data,
                        stochastic_optimizer type, size_t epochs, scalar_t alpha0, scalar_t beta,
                        trainer_result_t& result, thread_pool_t::mutex_t& mutex)
                {
                        samples_t tsamples = data.m_tsampler.get();
                        samples_t vsamples = data.m_vsampler.get();

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
                }
        }

        trainer_result_t stochastic_train(
                const model_t& model, const task_t& task, const sampler_t& tsampler, const sampler_t& vsampler, size_t nthreads,
                const loss_t& loss, const string_t& criterion,
                stochastic_optimizer optimizer, size_t epochs)
        {
                const auto op = [&] (scalar_t lambda)
                {
                        // prepare workers
                        thread_pool_t wpool(nthreads);
                        thread_pool_t::mutex_t mutex;

                        const size_t iterations = epochs * tsampler.size();             // SGD iterations
                        const scalar_t beta = std::pow(0.01, 1.0 / iterations);         // Learning rate decay rate

                        vector_t x0;
                        model.save_params(x0);

                        trainer_result_t result;

                        // tune the learning rate
                        const scalars_t alphas = { 0.001, 0.010, 0.100 };
                        for (scalar_t alpha : alphas)
                        {
                                wpool.enqueue([=, &task, &loss, &model, &x0, &result, &mutex]()
                                {
                                        accumulator_t lacc(model, 1, criterion, criterion_t::type::value, lambda);
                                        accumulator_t gacc(model, 1, criterion, criterion_t::type::vgrad, lambda);

                                        trainer_data_t data(task, tsampler, vsampler, loss, x0, lacc, gacc);

                                        detail::stochastic_train(data, optimizer, epochs, alpha, beta, result, mutex);
                                });
                        }

                        wpool.wait();

                        // OK
                        return result;
                };

                // tune the regularization factor (if needed)
                return detail::tune(op, criterion);
        }

        stochastic_trainer_t::stochastic_trainer_t(const string_t& parameters)
                :       trainer_t(parameters,
                                  "parameters: opt=sg[,sga,sia],epoch=16[1,1024]")
        {
        }

        trainer_result_t stochastic_trainer_t::train(
                const task_t& task, const fold_t& fold, const loss_t& loss, size_t nthreads, const string_t& criterion,
                model_t& model) const
        {
                if (fold.second != protocol::train)
                {
                        log_error() << "stochastic trainer: can only train models with training samples!";
                        return trainer_result_t();
                }

                // initialize the model
                model.resize(task, true);
                model.random_params();

                // prune training & validation data
                sampler_t tsampler(task);
                tsampler.setup(fold).setup(sampler_t::atype::annotated);

                sampler_t vsampler(task);
                tsampler.split(90, vsampler);

                if (tsampler.empty() || vsampler.empty())
                {
                        log_error() << "stochastic trainer: no annotated training samples!";
                        return trainer_result_t();
                }

                // parameters
                const size_t epochs = math::clamp(text::from_params<size_t>(configuration(), "epoch", 16), 1, 1024);

                const stochastic_optimizer optimizer = text::from_string<stochastic_optimizer>
                        (text::from_params<string_t>(configuration(), "opt", "sg"));

                // train the model
                const trainer_result_t result = ncv::stochastic_train(
                        model, task, tsampler, vsampler, nthreads,
                        loss, criterion, optimizer, epochs);

                log_info() << "optimum [train = " << result.m_opt_state.m_tvalue << "/" << result.m_opt_state.m_terror
                           << ", valid = " << result.m_opt_state.m_vvalue << "/" << result.m_opt_state.m_verror
                           << ", epoch = " << result.m_opt_epoch
                           << ", config = " << text::concatenate(result.m_opt_config, "/")
                           << "].";

                // OK
                if (result.valid())
                {
                        model.load_params(result.m_opt_params);
                }
                return result;
        }
}
