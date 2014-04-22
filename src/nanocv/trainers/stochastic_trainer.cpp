#include "stochastic_trainer.h"
#include "common/timer.h"
#include "common/logger.h"
#include "common/math.hpp"
#include "common/usampler.hpp"
#include "common/random.hpp"
#include "common/thread_pool.h"

namespace ncv
{
        /////////////////////////////////////////////////////////////////////////////////////////

        stochastic_trainer_t::stochastic_trainer_t(const string_t& parameters)
                :       trainer_t(parameters, "stochastic trainer, parameters: opt=sgd[,asgd],epoch=16[1,1024]"),
                        m_optimizer(text::from_params<string_t>(parameters, "opt", "sgd")),
                        m_epochs(text::from_params<size_t>(parameters, "epoch", 16))
        {
                m_epochs = math::clamp(m_epochs, 1, 1024);
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

        static void sgd_train(
                const task_t& task, samples_t& tsamples, const samples_t& vsamples, const loss_t& loss,
                size_t epochs, scalar_t alpha0, scalar_t beta, bool asgd,
                const model_t& model, trainer_state_t& state, thread_pool_t::mutex_t& mutex)
        {
                trainer_data_t ldata(model, trainer_data_t::type::value);
                trainer_data_t gdata(model, trainer_data_t::type::vgrad);

                random_t<size_t> xrng(0, tsamples.size());
                rnd_t xrnd(xrng);

                // (weighted-average) stochastic gradient descent
                timer_t timer;

                vector_t x = model.params();

                scalar_t alpha = alpha0;
                vector_t avgx = x;
                scalar_t sumb = 1.0 / alpha;

                for (size_t e = 0; e < epochs; e ++)
                {
                        std::random_shuffle(tsamples.begin(), tsamples.end(), xrnd);

                        // average stochastic gradient descent
                        if (asgd)
                        {
                                for (size_t i = 0; i < tsamples.size(); i ++, alpha *= beta)
                                {
                                        gdata.clear(x);
                                        gdata.update(task, tsamples[i], loss);

                                        x.noalias() -= alpha * gdata.vgrad();

                                        const scalar_t b = 1.0 / alpha;
                                        avgx = (avgx * sumb + x * b) / (sumb + b);
                                        sumb = sumb + b;
                                }
                        }

                        // stochastic gradient descent
                        else
                        {
                                for (size_t i = 0; i < tsamples.size(); i ++, alpha *= beta)
                                {
                                        gdata.clear(x);
                                        gdata.update(task, tsamples[i], loss);

                                        x.noalias() -= alpha * gdata.vgrad();
                                }
                        }

                        const vector_t xparam = asgd ? avgx : x;

                        // evaluate training samples
                        ldata.clear(xparam);
                        ldata.update_st(task, tsamples, loss);
                        const scalar_t tvalue = ldata.value();
                        const scalar_t terror = ldata.error();

                        // evaluate validation samples
                        ldata.clear(xparam);
                        ldata.update_st(task, vsamples, loss);
                        const scalar_t vvalue = ldata.value();
                        const scalar_t verror = ldata.error();

                        // OK, update the optimum solution
                        const thread_pool_t::lock_t lock(mutex);

                        if (state.update(xparam, tvalue, terror, vvalue, verror, 0.0))
                        {
                                log_info()
                                << "[train* = " << state.m_tvalue << "/" << state.m_terror
                                << ", valid* = " << state.m_vvalue << "/" << state.m_verror
                                << ", rate = " << alpha << "/" << alpha0
                                << ", epoch = " << e << "/" << epochs
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
                const samples_t samples = ncv::prune_annotated(task, task.samples(fold));
                if (samples.empty())
                {
                        log_error() << "stochastic trainer: no annotated training samples!";
                        return false;
                }

                samples_t tsamples, vsamples;
                ncv::uniform_split(samples, size_t(90), random_t<size_t>(0, samples.size()), tsamples, vsamples);

                //
                if (    !text::iequals(m_optimizer, "asgd") &&
                        !text::iequals(m_optimizer, "sgd"))
                {
                        log_error() << "stochastic trainer: invalid optimization method <" << m_optimizer << ">!";
                        return false;
                }

                const bool asgd = text::iequals(m_optimizer, "asgd");

                // prepare workers
                thread_pool_t wpool(nthreads);
                thread_pool_t::mutex_t mutex;

                const size_t iterations = m_epochs * tsamples.size();           // SGD iterations
                const scalar_t beta = std::pow(0.01, 1.0 / iterations);         // Learning rate decay rate

                // optimum model parameters (to update)
                trainer_state_t state(model.n_parameters());
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
                                sgd_train(task, tsamples, vsamples, loss,
                                          m_epochs, alpha0, beta, asgd,
                                          model, state, mutex);
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
