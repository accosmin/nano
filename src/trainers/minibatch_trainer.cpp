#include "minibatch_trainer.h"
#include "core/logger.h"
#include "core/optimize.h"
#include "core/timer.h"
#include "core/text.h"
#include "core/math/clamp.hpp"

namespace ncv
{
        //-------------------------------------------------------------------------------------------------

        minibatch_trainer_t::minibatch_trainer_t(const string_t& params)
                :       m_optimizer(text::from_params<string_t>(params, "opt", "lbfgs")),
                        m_iterations(text::from_params<size_t>(params, "iter", 16)),
                        m_batchsize(text::from_params<size_t>(params, "batch", 1024)),
                        m_epochs(text::from_params<size_t>(params, "epoch", 256)),
                        m_epsilon(1e-6)
        {
                m_iterations = math::clamp(m_iterations, 4, 128);
                m_batchsize = math::clamp(m_batchsize, 100, 10000);
                m_epochs = math::clamp(m_epochs, 8, 1024);
        }

        //-------------------------------------------------------------------------------------------------

        static void update(const optimize::result_t& result, timer_t& timer, size_t epoch, size_t epochs)
        {
                ncv::log_info() << "mini-batch trainer: state [epoch = " << epoch << "/" << epochs
                                << ", loss = " << result.optimum().f
                                << ", gradient = " << result.optimum().g.lpNorm<Eigen::Infinity>()
                                << ", calls = " << result.n_fval_calls() << "/" << result.n_grad_calls()
                                << "] updated in " << timer.elapsed() << ".";
                timer.start();
        }

        //-------------------------------------------------------------------------------------------------

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
                const samples_t samples = trainer_t::prune_annotated(task, task.samples(fold));
                if (samples.empty())
                {
                        log_error() << "mini-batch trainer: no annotated training samples!";
                        return false;
                }

                // current mini-batch of samples
                samples_t bsamples(m_batchsize);

                // optimization problem: size
                auto opt_fn_size = [&] ()
                {
                        return model.n_parameters();
                };

                // optimization problem: function value
                auto opt_fn_fval = [&] (const vector_t& x)
                {
                        model.load_params(x);
                        return trainer_t::value_mt(task, bsamples, loss, model);
                };

                // optimization problem: function value & gradient
                auto opt_fn_fval_grad = [&] (const vector_t& x, vector_t& gx)
                {
                        model.load_params(x);
                        return trainer_t::vgrad_mt(task, bsamples, loss, model, gx);
                };

                const optimize::problem_t problem(opt_fn_size, opt_fn_fval, opt_fn_fval_grad);

                timer_t timer;

                optimize::result_t res(model.n_parameters());

                vector_t x(model.n_parameters());
                model.save_params(x);

                // mini-batch optimization
                for (size_t epoch = 0; epoch < m_epochs; epoch ++)
                {
                        // select random mini-batch
                        random_t<size_t> die(0, samples.size() - 1);
                        for (size_t i = 0; i < m_batchsize; i ++)
                        {
                                bsamples[i] = samples[die()];
                        }

                        // optimize the model                                                
                        const auto updater = std::bind(update, _1, std::ref(timer), epoch + 1, m_epochs);

                        optimize::result_t bres;
                        if (text::iequals(m_optimizer, "lbfgs"))
                        {
                                bres = optimize::lbfgs(problem, x, m_iterations, m_epsilon, 6, updater);
                        }
                        else if (text::iequals(m_optimizer, "cgd"))
                        {
                                bres = optimize::cgd(problem, x, m_iterations, m_epsilon, updater);
                        }
                        else if (text::iequals(m_optimizer, "gd"))
                        {
                                bres = optimize::gd(problem, x, m_iterations, m_epsilon, updater);
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
                           << "], speed = [" << res.speed().avg() << " +/- " << res.speed().stdev() << "].";

                return true;
        }

        //-------------------------------------------------------------------------------------------------
}
