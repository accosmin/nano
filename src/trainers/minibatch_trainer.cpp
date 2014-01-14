#include "minibatch_trainer.h"
#include "util/timer.h"
#include "util/logger.h"
#include "util/math.hpp"
#include "util/usampler.hpp"
#include "thread/thread_pool.h"
#include "text.h"

namespace ncv
{
        /////////////////////////////////////////////////////////////////////////////////////////

        minibatch_trainer_t::minibatch_trainer_t(const string_t& params)
                :     m_optimizer(text::from_params<string_t>(params, "opt", "cgd")),
                      m_iterations(text::from_params<size_t>(params, "iters", 16)),
                      m_epsilon(text::from_params<scalar_t>(params, "eps", 1e-6)),
                      m_batch(text::from_params<size_t>(params, "batch", 256)),
                      m_epochs(text::from_params<size_t>(params, "epoch", 16))

        {
                m_iterations = math::clamp(m_iterations, 4, 256);
                m_epsilon = math::clamp(m_epsilon, 1e-8, 1e-3);
                m_batch = math::clamp(m_epochs, 256, 16 * 1024);
                m_epochs = math::clamp(m_epochs, 1, 256);
        }

        /////////////////////////////////////////////////////////////////////////////////////////

        bool minibatch_trainer_t::train(
                const task_t& task, const fold_t& fold, const loss_t& loss, size_t nthreads, model_t& model) const
        {
                if (fold.second != protocol::train)
                {
                        log_error() << "minibatch trainer: cannot only train models with training samples!";
                        return false;
                }

                // initialize the model
                model.resize(task);
                model.random_params();

                // prune training & validation data
                const samples_t samples = ncv::prune_annotated(task, task.samples(fold));
                if (samples.empty())
                {
                        log_error() << "minibatch trainer: no annotated training samples!";
                        return false;
                }

                samples_t tsamples, vsamples;
                ncv::uniform_split(samples, size_t(80), random_t<size_t>(0, samples.size()), tsamples, vsamples);

                const size_t tsize = m_batch;
                const size_t vsize = 4 * tsize;

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

                                // optimize starting from the current optimum solution
                                const rmodel_t nmodel = model.clone();
                                {
                                        const thread_pool_t::lock_t lock(mutex);
                                        nmodel->load_params(opt_state.m_params);
                                }

                                trainer_state_t state(model.n_parameters());

                                timer_t timer;
                                ncv::train(task, ntsamples, nvsamples, loss,
                                           m_optimizer, m_epsilon, m_iterations, 1,     // 1 thread!
                                           *nmodel, state, false);

                                // OK, update the optimum solution
                                {
                                        const thread_pool_t::lock_t lock(mutex);
                                        opt_state.update(state);

                                        log_info() << "minibatch trainer: [train* = " << opt_state.m_tvalue << "/" << opt_state.m_terror
                                                   << ", valid* = " << opt_state.m_vvalue << "/" << opt_state.m_verror
                                                   << "] done in " << timer.elapsed() << ".";
                                }
                        });
                }

                wpool.wait();

                model.load_params(opt_state.m_params);

                log_info() << "minibatch trainer: [train* = " << opt_state.m_tvalue << "/" << opt_state.m_terror
                           << ", valid* = " << opt_state.m_vvalue << "/" << opt_state.m_verror << "].";

                // OK
                return true;
        }

        /////////////////////////////////////////////////////////////////////////////////////////
}
