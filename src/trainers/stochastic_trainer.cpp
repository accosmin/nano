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
                :       m_gamma(text::from_params<scalar_t>(params, "gamma", 1e-2)),
                        m_beta(text::from_params<scalar_t>(params, "beta", 0.999)),
                        m_batch(text::from_params<size_t>(params, "batch", 1024)),
                        m_epochs(text::from_params<size_t>(params, "epoch", 16))
        {
                m_gamma = math::clamp(m_gamma, 1e-3, 1e-1);
                m_beta = math::clamp(m_beta, 0.5, 1.0);
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
                ncv::uniform_split(samples, size_t(80), random_t<size_t>(0, samples.size()), tsamples, vsamples);

                const size_t tsize = m_batch;
                const size_t vsize = 4 * tsize;
                const size_t iters = m_batch;

                // optimum model parameters
                trainer_state_t opt_state(model.n_parameters());
                opt_state.m_params = model.params();

                // prepare workers
                ncv::thread_pool_t wpool(nthreads);
                const size_t n_workers = wpool.n_workers();

                thread_pool_t::mutex_t mutex;

                // optimize the model by exploring the parameter space with multiple workers
                for (size_t e = 0; e < m_epochs * n_workers; e ++)
                {
                        wpool.enqueue([&]()
                        {
                                const random_t<size_t> trng(0, tsamples.size());

                                const random_t<size_t> vrng(0, vsamples.size());
                                const samples_t ntsamples = ncv::uniform_sample(tsamples, tsize, trng);
                                const samples_t nvsamples = ncv::uniform_sample(vsamples, vsize, vrng);

                                // current optimum solution
                                vector_t x;
                                {
                                        const thread_pool_t::lock_t lock(mutex);
                                        x = opt_state.m_params;
                                }

                                // stochastic gradient descent
                                timer_t timer;

                                trainer_data_skipgrad_t ldata(model);
                                trainer_data_withgrad_t gdata(model);

                                random_t<size_t> rng(0, ntsamples.size());

                                scalar_t alpha = m_gamma;
                                for (size_t i = 0; i < iters; i ++, alpha *= m_beta)
                                {
                                        gdata.load_params(x);
                                        gdata.update(task, ntsamples[rng() % ntsamples.size()], loss);

                                        x -= alpha * gdata.vgrad();
                                }

                                // training samples: loss value
                                ldata.load_params(x);
                                ldata.update_st(task, ntsamples, loss);
                                const scalar_t tvalue = ldata.value();
                                const scalar_t terror = ldata.error();

                                // validation samples: loss value
                                ldata.load_params(x);
                                ldata.update_st(task, nvsamples, loss);
                                const scalar_t vvalue = ldata.value();
                                const scalar_t verror = ldata.error();

                                // OK, update the optimum solution
                                {
                                        const thread_pool_t::lock_t lock(mutex);
                                        opt_state.update(x, tvalue, terror, vvalue, verror);

                                        log_info() << "[train* = "
                                                   << opt_state.m_tvalue << "/" << opt_state.m_terror
                                                   << ", valid* = " << opt_state.m_vvalue << "/" << opt_state.m_verror
                                                   << ", rate = " << m_gamma << " -> " << alpha
                                                   << "] done in " << timer.elapsed() << ".";
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
