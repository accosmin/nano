#include "stochastic_trainer.h"
#include "common/timer.h"
#include "common/logger.h"
#include "common/math.hpp"
#include "common/uniform.hpp"
#include "common/random.hpp"
#include "common/thread_pool.h"
#include "sampler.h"
#include "accumulator.h"

namespace ncv
{
        /////////////////////////////////////////////////////////////////////////////////////////

        stochastic_trainer_t::stochastic_trainer_t(const string_t& parameters)
                :       trainer_t(parameters,
                                  "stochastic trainer, "\
                                  "parameters: opt=sg[,sga,sia],epoch=16[1,1024]")
        {
        }

        /////////////////////////////////////////////////////////////////////////////////////////

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

        /////////////////////////////////////////////////////////////////////////////////////////

        static void sgd_train(
                const task_t& task, samples_t& tsamples, const samples_t& vsamples, const loss_t& loss,
                size_t epochs, scalar_t alpha0, scalar_t beta, stochastic_optimizer type, scalar_t lambda,
                const model_t& model, trainer_state_t& state, thread_pool_t::mutex_t& mutex)
        {
                accumulator_t ldata(model, accumulator_t::type::value, lambda);
                accumulator_t gdata(model, accumulator_t::type::vgrad, lambda);

                random_t<size_t> xrng(0, tsamples.size());
                rnd_t xrnd(xrng);

                // (weighted-average) stochastic gradient descent
                timer_t timer;

                vector_t x = model.params();
                vector_t xparam = x;

                vector_t xavg = x;
                vector_t gavg(x.size());
                gavg.setZero();

                scalar_t alpha = alpha0;
                scalar_t sumb = 1.0 / alpha;

                for (size_t e = 0; e < epochs; e ++)
                {
                        std::random_shuffle(tsamples.begin(), tsamples.end(), xrnd);

                        switch (type)
                        {
                        case stochastic_optimizer::SG:
                                for (size_t i = 0; i < tsamples.size(); i ++, alpha *= beta)
                                {
                                        gdata.reset(x);
                                        gdata.update(task, tsamples[i], loss);

                                        x.noalias() -= alpha * gdata.vgrad();
                                }
                                xparam = x;
                                break;

                        case stochastic_optimizer::SGA:
                                for (size_t i = 0; i < tsamples.size(); i ++, alpha *= beta)
                                {
                                        gdata.reset(x);
                                        gdata.update(task, tsamples[i], loss);

                                        const vector_t g = gdata.vgrad();

                                        const scalar_t b = 1.0 / alpha;
                                        gavg = (gavg * sumb + g * b) / (sumb + b);
                                        sumb = sumb + b;

                                        x.noalias() -= alpha * gavg;
                                }
                                xparam = xavg;
                                break;

                        case stochastic_optimizer::SIA:
                        default:
                                for (size_t i = 0; i < tsamples.size(); i ++, alpha *= beta)
                                {
                                        gdata.reset(x);
                                        gdata.update(task, tsamples[i], loss);

                                        x.noalias() -= alpha * gdata.vgrad();

                                        const scalar_t b = 1.0 / alpha;
                                        xavg = (xavg * sumb + x * b) / (sumb + b);
                                        sumb = sumb + b;
                                }
                                xparam = x;
                                break;
                        }

                        // evaluate training samples
                        ldata.reset(xparam);
                        ldata.update(task, tsamples, loss);
                        const scalar_t tvalue = ldata.value();
                        const scalar_t terror = ldata.error();

                        // evaluate validation samples
                        ldata.reset(xparam);
                        ldata.update(task, vsamples, loss);
                        const scalar_t vvalue = ldata.value();
                        const scalar_t verror = ldata.error();

                        // OK, update the optimum solution
                        const thread_pool_t::lock_t lock(mutex);

                        if (state.update(xparam, tvalue, terror, vvalue, verror,
                                         ldata.lambda(), e * tsamples.size(), e * tsamples.size()))
                        {
                                log_info()
                                << "[rate = " << alpha << "/" << alpha0
                                << ", epoch = " << e << "/" << epochs
                                << ", train* = " << state.m_tvalue << "/" << state.m_terror
                                << ", valid* = " << state.m_vvalue << "/" << state.m_verror                                   
                                << ", lambda* = " << ldata.lambda() << "/" << state.m_lambda
                                << "] done in " << timer.elapsed() << ".";
                        }
                }
        }

        /////////////////////////////////////////////////////////////////////////////////////////

        bool stochastic_trainer_t::train(
                const task_t& task, const fold_t& fold, const loss_t& loss, size_t nthreads, model_t& model) const
        {
                if (fold.second != protocol::train)
                {
                        log_error() << "stochastic trainer: cannot only train models with training samples!";
                        return false;
                }

                // initialize the model
                model.resize(task, true);
                model.random_params();

                // prune training & validation data
                sampler_t sampler(task);
                sampler.setup(fold).setup(sampler_t::atype::annotated);

                const samples_t samples = sampler.get();
                if (samples.empty())
                {
                        log_error() << "stochastic trainer: no annotated training samples!";
                        return false;
                }

                samples_t tsamples, vsamples;
                ncv::uniform_split(samples, size_t(90), random_t<size_t>(0, samples.size()), tsamples, vsamples);

                // parameters
                const size_t epochs = math::clamp(text::from_params<size_t>(configuration(), "epoch", 16), 1, 1024);
                const size_t iterations = epochs * tsamples.size();             // SGD iterations
                const scalar_t beta = std::pow(0.01, 1.0 / iterations);         // Learning rate decay rate

                const stochastic_optimizer optimizer = text::from_string<stochastic_optimizer>
                        (text::from_params<string_t>(configuration(), "opt", "sg"));

                // prepare workers
                thread_pool_t wpool(nthreads);
                thread_pool_t::mutex_t mutex;

                // optimum model parameters (to update)
                trainer_state_t state(model.psize());
                state.m_params = model.params();

                // tune the learning rate
                const scalar_t max_alpha = 1e-1;
                const scalar_t min_alpha = 1e-3;
                const scalar_t var_alpha = std::exp((std::log(max_alpha) - std::log(min_alpha))
                                           / std::min(size_t(8), wpool.n_workers()));

                for (scalar_t alpha0 = min_alpha; alpha0 <= max_alpha; alpha0 *= var_alpha)
                {
                        wpool.enqueue([=, &model, &task, &tsamples, &vsamples, &loss, &state, &mutex]()
                        {
                                // tune the L2-regularization scale
                                const scalars_t lambdas = { 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1.0 };
                                for (scalar_t lambda : lambdas)
                                {
                                        sgd_train(task, tsamples, vsamples, loss,
                                                  epochs, alpha0, beta, optimizer, lambda,
                                                  model, state, mutex);
                                }
                        });
                }

                wpool.wait();

                model.load_params(state.m_params);

                log_info() << "[train* = " << state.m_tvalue << "/" << state.m_terror
                           << ", valid* = " << state.m_vvalue << "/" << state.m_verror << "].";

                // OK
                return true;
        }

        /////////////////////////////////////////////////////////////////////////////////////////
}
