#include "minibatch_trainer.h"
#include "core/timer.h"
#include "core/text.h"
#include "core/math/clamp.hpp"
#include "core/logger.h"

namespace ncv
{
        /////////////////////////////////////////////////////////////////////////////////////////

        minibatch_trainer_t::minibatch_trainer_t(const string_t& params)
                :       m_optimizer(text::from_params<string_t>(params, "opt", "lbfgs")),
                        m_iterations(text::from_params<size_t>(params, "iter", 16)),
                        m_batchsize(text::from_params<size_t>(params, "batch", 1024)),
                        m_epochs(text::from_params<size_t>(params, "epoch", 256)),
                        m_epsilon(1e-6),
                        m_sampling(text::from_params<string_t>(params, "sample", "lwei"))
        {
                m_iterations = math::clamp(m_iterations, 4, 128);
                m_batchsize = math::clamp(m_batchsize, 100, 10000);
                m_epochs = math::clamp(m_epochs, 8, 1024);
        }

        /////////////////////////////////////////////////////////////////////////////////////////

        minibatch_trainer_t::lvalues_t minibatch_trainer_t::make_lvalues(const samples_t& samples,
                const task_t& task, const loss_t& loss, const model_t& model) const
        {
                lvalues_t lvalues;

                for (size_t i = 0; i < samples.size(); i ++)
                {
                        lvalues.emplace_back(ncv::lvalue(task, samples[i], loss, model), i);
                }

                return lvalues;
        }

        /////////////////////////////////////////////////////////////////////////////////////////

        samples_t minibatch_trainer_t::make_samples(const samples_t& samples, const lvalues_t& lvalues) const
        {
                samples_t bsamples;

                for (size_t i = 0; i < lvalues.size(); i ++)
                {
                        const sample_t& sample = samples[lvalues[i].second];
                        bsamples.emplace_back(sample);
                }

                return bsamples;
        }

        /////////////////////////////////////////////////////////////////////////////////////////

        samples_t minibatch_trainer_t::rand(const samples_t& samples) const
        {
                samples_t bsamples;

                for (size_t i = 0; i < m_batchsize; i ++)
                {
                        random_t<size_t> die(0, samples.size() - 1);

                        const sample_t& sample = samples[die()];
                        bsamples.emplace_back(sample);
                }

                return bsamples;        
        }

        /////////////////////////////////////////////////////////////////////////////////////////

        samples_t minibatch_trainer_t::lmax(const samples_t& bsamples, const samples_t& samples,
                const task_t& task, const loss_t& loss, const model_t& model) const
        {
                samples_t qsamples = rand(samples);
                qsamples.insert(qsamples.end(), bsamples.begin(), bsamples.end());

                lvalues_t qlvalues = make_lvalues(qsamples, task, loss, model);

                std::sort(qlvalues.begin(), qlvalues.end());
                const lvalues_t blvalues(qlvalues.begin() + m_batchsize, qlvalues.end());

                return make_samples(qsamples, blvalues);
        }

        /////////////////////////////////////////////////////////////////////////////////////////

        samples_t minibatch_trainer_t::lwei(const samples_t& bsamples, const samples_t& samples,
                const task_t& task, const loss_t& loss, const model_t& model) const
        {
                // fixme: double check this implementation!

                samples_t qsamples = rand(samples);
                qsamples.insert(qsamples.end(), bsamples.begin(), bsamples.end());

                lvalues_t qlvalues = make_lvalues(qsamples, task, loss, model);

                const scalar_t lsum = std::accumulate(qlvalues.begin(), qlvalues.end(), 0.0,
                                      [] (scalar_t sum, const lvalue_t& lv) { return sum + lv.first; });

                const scalar_t pdiv = m_batchsize / (lsum + std::numeric_limits<scalar_t>::epsilon());

                lvalues_t blvalues;
                for (size_t i = 0; i < qlvalues.size(); i ++)
                {
                        for (   scalar_t prob = qlvalues[i].first * pdiv;
                                prob > std::numeric_limits<scalar_t>::epsilon();
                                prob -= 1.0)
                        {
                                random_t<scalar_t> die(0.0, 1.0);
                                if (prob > die())
                                {
                                        blvalues.push_back(qlvalues[i]);
                                }
                        }
                }

                return make_samples(qsamples, blvalues);
        }

        /////////////////////////////////////////////////////////////////////////////////////////

        bool minibatch_trainer_t::train(const task_t& task, const fold_t& fold, const loss_t& loss, model_t& model) const
        {
                if (fold.second != protocol::train)
                {
                        log_error() << "mini-batch trainer: cannot only train models with training samples!";
                        return false;
                }

                // initialize the model
                model.resize(task);
                model.random_params();

                // prune training data
                const samples_t samples = ncv::prune_annotated(task, task.samples(fold));
                if (samples.empty())
                {
                        log_error() << "mini-batch trainer: no annotated training samples!";
                        return false;
                }

                // current mini-batch of samples
                samples_t bsamples = rand(samples);

                // optimization problem: size
                auto fn_size = [&] ()
                {
                        return model.n_parameters();
                };

                // optimization problem: function value
                auto fn_fval = [&] (const vector_t& x)
                {
                        model.load_params(x);
                        return ncv::lvalue_mt(task, bsamples, loss, model);
                };

                // optimization problem: function value & gradient
                auto fn_grad = [&] (const vector_t& x, vector_t& gx)
                {
                        model.load_params(x);
                        return ncv::lvgrad_mt(task, bsamples, loss, model, gx);
                };

                // optimization problem: logging
                auto fn_wlog = [] (const string_t& message)
                {
                        log_warning() << message;
                };
                auto fn_elog = [] (const string_t& message)
                {
                        log_error() << message;
                };
                auto fn_ulog = [] (const opt_result_t& result, timer_t& timer, size_t epoch, size_t epochs)
                {
                        log_info() << "mini-batch trainer: state [epoch = " << epoch << "/" << epochs
                                << ", loss = " << result.optimum().f
                                << ", gradient = " << result.optimum().g.lpNorm<Eigen::Infinity>()
                                << ", calls = " << result.n_fval_calls() << " fun/ " << result.n_grad_calls()
                                << " grad] updated in " << timer.elapsed() << ".";
                        timer.start();
                };

                // assembly optimization problem
                const opt_problem_t problem(fn_size, fn_fval, fn_grad);
                opt_result_t res(model.n_parameters());

                timer_t timer;

                vector_t x(model.n_parameters());
                model.save_params(x);

                // mini-batch optimization
                for (size_t epoch = 0; epoch < m_epochs; epoch ++)
                {                        
                        // update the current mini-batch
                        if (text::iequals(m_sampling, "once"))
                        {
                        }
                        else if (text::iequals(m_sampling, "rand"))
                        {
                                bsamples = rand(samples);
                        }
                        else if (text::iequals(m_sampling, "lmax"))
                        {
                                bsamples = lmax(bsamples, samples, task, loss, model);
                        }
                        else if (text::iequals(m_sampling, "lwei"))
                        {
                                bsamples = lwei(bsamples, samples, task, loss, model);
                        }
                        else
                        {
                                log_error() << "mini-batch trainer: invalid sampling method <" << m_sampling << ">!";
                                return false;
                        }

                        const auto fn_ulog_ref = std::bind(fn_ulog, _1, std::ref(timer), epoch + 1, m_epochs);
                        const scalar_t eps = m_epsilon;
                        const size_t iters = m_iterations;

                        // optimize the model
                        opt_result_t bres;
                        if (text::iequals(m_optimizer, "lbfgs"))
                        {
                                bres = optimizer_t::lbfgs(problem, x, iters, eps, 6, fn_wlog, fn_elog, fn_ulog_ref);
                        }
                        else if (text::iequals(m_optimizer, "cgd"))
                        {
                                bres = optimizer_t::cgd(problem, x, iters, eps, fn_wlog, fn_elog, fn_ulog_ref);
                        }
                        else if (text::iequals(m_optimizer, "gd"))
                        {
                                bres = optimizer_t::gd(problem, x, iters, eps, fn_wlog, fn_elog, fn_ulog_ref);
                        }
                        else
                        {
                                log_error() << "mini-batch trainer: invalid optimization method <" << m_optimizer << ">!";
                                return false;
                        }

                        // update the model
                        x = bres.optimum().x;
                        res.update(bres);
                }

                // update the model
                model.load_params(res.optimum().x);

                // OK
                log_info() << "mini-batch trainer: optimum [loss = " << res.optimum().f
                           << ", gradient = " << res.optimum().g.norm()
                           << ", calls = " << res.n_fval_calls() << "/" << res.n_grad_calls()
                           << "], iterations = [" << res.iterations() << "/" << m_iterations
                           << "].";

                return true;
        }

        /////////////////////////////////////////////////////////////////////////////////////////
}
