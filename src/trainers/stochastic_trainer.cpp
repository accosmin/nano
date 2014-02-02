#include "stochastic_trainer.h"
#include "util/timer.h"
#include "util/logger.h"
#include "util/math.hpp"
#include "util/usampler.hpp"
#include "text.h"
#include "trainer_data.h"
#include "trainer_state.h"

namespace ncv
{
        /////////////////////////////////////////////////////////////////////////////////////////

        stochastic_trainer_t::stochastic_trainer_t(const string_t& params)
                :       m_optimizer(text::from_params<string_t>(params, "opt", "sgd")),
                        m_epochs(text::from_params<size_t>(params, "epoch", 16))
        {
                m_epochs = math::clamp(m_epochs, 1, 1024);
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
                model.resize(task);
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
                ncv::thread_pool_t wpool(nthreads);
                thread_pool_t::mutex_t mutex;

                const size_t iterations = m_epochs * tsamples.size();   // SGD iterations
                const scalar_t beta = std::pow(0.01, 1.0 / iterations); // Learning rate range: lamba -> lambda/100

                // optimum model parameters (to update)
                trainer_state_t state(model.n_parameters());
                state.m_params = model.params();

                // tune the learning rate
                const scalar_t max_alpha = 1e-1;
                const scalar_t min_alpha = 1e-3;
                const scalar_t var_alpha = std::exp((std::log(max_alpha) - std::log(min_alpha))
                                           / (4.0 * wpool.n_workers()));

                for (scalar_t alpha0 = min_alpha; alpha0 <= max_alpha; alpha0 *= var_alpha)
                {
                        wpool.enqueue([=, &model, &task, &tsamples, &vsamples, &loss, &state, &mutex]()
                        {
                                trainer_data_skipgrad_t ldata(model);
                                trainer_data_withgrad_t gdata(model);

                                random_t<size_t> xrng(0, tsamples.size());

                                // (weighted-average) stochastic gradient descent
                                timer_t timer;

                                vector_t x = model.params();

                                scalar_t alpha = alpha0;
                                vector_t avgx = x;
                                scalar_t sumb = 1.0 / alpha;

                                for (size_t i = 0; i < iterations; i ++, alpha *= beta)
                                {
                                        gdata.load_params(x);
                                        gdata.update(task, tsamples[xrng() % tsamples.size()], loss);

                                        x -= alpha * gdata.vgrad();

                                        // running weighted average update
                                        if (asgd)
                                        {
                                                const scalar_t beta = 1.0 / alpha;
                                                avgx = (avgx * sumb + x * beta) / (sumb + beta);
                                                sumb = sumb + beta;
                                        }

                                        // check from time to time its performance
                                        if ((i % tsamples.size()) == 0 || (i + 1) == iterations)
                                        {
                                                const vector_t xparam = asgd ? avgx : x;

                                                // training samples: loss value
                                                ldata.load_params(xparam);
                                                ldata.update_st(task, tsamples, loss);
                                                const scalar_t tvalue = ldata.value();
                                                const scalar_t terror = ldata.error();

                                                // validation samples: loss value
                                                ldata.load_params(xparam);
                                                ldata.update_st(task, vsamples, loss);
                                                const scalar_t vvalue = ldata.value();
                                                const scalar_t verror = ldata.error();

                                                // OK, update the optimum solution
                                                const thread_pool_t::lock_t lock(mutex);

                                                const size_t epoch = (i / tsamples.size()) + 1;

                                                if (state.update(xparam, tvalue, terror, vvalue, verror))
                                                {
                                                        log_info()
                                                        << "[train* = " << state.m_tvalue << "/" << state.m_terror
                                                        << ", valid* = " << state.m_vvalue << "/" << state.m_verror
                                                        << ", rate = " << alpha << "/" << alpha0
                                                        << ", epoch = " << epoch << "/" << m_epochs
                                                        << "] done in " << timer.elapsed() << ".";
                                                }
                                        }
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
