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
                :       m_optimizer(text::from_params<string_t>(params, "opt", "asgd")),
                        m_alpha(text::from_params<scalar_t>(params, "alpha", 1e-2)),
                        m_batch(text::from_params<size_t>(params, "batch", 1024)),
                        m_epochs(text::from_params<size_t>(params, "epoch", 4))
        {
                m_alpha = math::clamp(m_alpha, 1e-3, 1e-1);
                m_batch = math::clamp(m_batch, 256, 16 * 1024);
                m_epochs = math::clamp(m_epochs, 1, 256);
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
                const size_t maxage = 8;                                // Stop if no improvement in the last X epochs

                // optimum model parameters (to update)
                trainer_state_t opt_state(model.n_parameters());
                opt_state.m_params = model.params();

                // optimize the model by exploring the parameter space with multiple workers
                for (size_t n = 0; n < nthreads; n ++)
                {
                        wpool.enqueue([&]()
                        {
                                trainer_data_skipgrad_t ldata(model);
                                trainer_data_withgrad_t gdata(model);

                                const size_t ntsize = m_batch;
                                const size_t nvsize = m_batch * 4;

                                random_t<size_t> trng(0, ntsize);
                                random_t<size_t> vrng(0, nvsize);
                                random_t<size_t> xrng(0, tsamples.size());

                                // (weighted-average) stochastic gradient descent
                                timer_t timer;

                                vector_t x = model.params();

                                scalar_t alpha = m_alpha;
                                vector_t avgx = x;
                                scalar_t sumb = 1.0 / alpha;

                                for (size_t i = 0, ia = 0; i < iterations && ia < maxage; i ++, alpha *= beta)
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
                                        if ((i % m_batch) == 0 || (i + 1) == iterations)
                                        {
                                                const vector_t xparam = asgd ? avgx : x;

                                                // training samples: loss value
                                                ldata.load_params(xparam);
                                                ldata.update_st(task, ncv::uniform_sample(tsamples, ntsize, trng), loss);
                                                const scalar_t tvalue = ldata.value();
                                                const scalar_t terror = ldata.error();

                                                // validation samples: loss value
                                                ldata.load_params(xparam);
                                                ldata.update_st(task, ncv::uniform_sample(vsamples, nvsize, vrng), loss);
                                                const scalar_t vvalue = ldata.value();
                                                const scalar_t verror = ldata.error();

                                                // OK, update the optimum solution
                                                {
                                                        const thread_pool_t::lock_t lock(mutex);

                                                        if (opt_state.update(xparam, tvalue, terror, vvalue, verror))
                                                        {
                                                                ia = 0;

                                                                log_info() << "[train* = "
                                                                           << opt_state.m_tvalue << "/" << opt_state.m_terror
                                                                           << ", valid* = "
                                                                           << opt_state.m_vvalue << "/" << opt_state.m_verror
                                                                           << ", rate = " << alpha
                                                                           << ", thread = " << (n + 1) << "/" << nthreads
                                                                           << "] done in " << timer.elapsed() << ".";
                                                        }

                                                        else
                                                        {
                                                                ia ++;
                                                        }
                                                }
                                        }
                                }


                        });
                }

                wpool.wait();

                model.load_params(opt_state.m_params);

                log_info() << "[train* = " << opt_state.m_tvalue << "/" << opt_state.m_terror
                           << ", valid* = " << opt_state.m_vvalue << "/" << opt_state.m_verror << "].";

                // OK
                return true;
        }

        /////////////////////////////////////////////////////////////////////////////////////////
}
