#include "stochastic_trainer.h"
#include "core/logger.h"
#include "core/optimize.h"
#include "core/timer.h"
#include "core/random.hpp"
#include "core/text.h"
#include "core/clamp.hpp"

namespace ncv
{
        //-------------------------------------------------------------------------------------------------

        stochastic_trainer_t::stochastic_trainer_t(const string_t& params)
                :       m_optimizer(text::from_params<string_t>(params, "opt", "asgd")),
                        m_epochs(text::from_params<size_t>(params, "epochs", 4)),
                        m_epsilon(text::from_params<scalar_t>(params, "eps", 1e-6))
        {
                m_epochs = math::clamp(m_epochs, 1, 32);
                m_epsilon = math::clamp(m_epsilon, 1e-8, 1e-3);
        }

        //-------------------------------------------------------------------------------------------------

        static void update(const optimize::result_t& result, timer_t& timer)
        {
                ncv::log_info() << "stochastic trainer: state [loss = " << result.optimum().f
                                << ", gradient = " << result.optimum().g.lpNorm<Eigen::Infinity>()
                                << ", calls = " << result.n_fval_calls() << "/" << result.n_grad_calls()
                                << "] updated in " << timer.elapsed() << ".";
                timer.start();
        }

        //-------------------------------------------------------------------------------------------------

        bool stochastic_trainer_t::train(const task_t& task, const fold_t& fold, const loss_t& loss, model_t& model) const
        {
                if (fold.second != protocol::train)
                {
                        log_error() << "stochastic trainer: cannot only train models with training samples!";
                        return false;
                }

                // initialize the model
                model.resize(task);
                model.random_params();

                // prune training data
                const samples_t& samples = trainer_t::prune_annotated(task, task.samples(fold));
                if (samples.empty())
                {
                        log_error() << "stochastic trainer: no annotated training samples!";
                        return false;
                }

                // optimization problem: size
                auto opt_fn_size = [&] ()
                {
                        return model.n_parameters();
                };

                // optimization problem: function value
                auto opt_fn_fval = [&] (const vector_t& x)
                {
                        model.load_params(x);

                        // NB: use all samples to evaluate the loss value!

                        return trainer_t::value(task, samples, loss, model);
                };

                // optimization problem: function value & gradient
                auto opt_fn_fval_grad = [&] (const vector_t& x, vector_t& gx)
                {
                        model.load_params(x);

                        // NB: use a random sample to evaluate the gradient!
                        random_t<size_t> rgen(0, samples.size() - 1);
                        samples_t gsamples;
                        for (size_t i = 0; i < 1; i ++)
                        {
                                const size_t index = rgen();
                                gsamples.push_back(samples[index]);
                        }

                        return trainer_t::vgrad(task, gsamples, loss, model, gx);
                };

                const optimize::problem_t problem(opt_fn_size, opt_fn_fval, opt_fn_fval_grad);

                // optimize the model
                vector_t x(model.n_parameters());
                model.save_params(x);

                timer_t timer;

                const auto updater = std::bind(update, _1, std::ref(timer));

                optimize::result_t res;
                if (text::iequals(m_optimizer, "asgd"))
                {
                        res = optimize::asgd(problem, x, m_epochs * samples.size(), m_epsilon, updater);
                }
                else if (text::iequals(m_optimizer, "sgd"))
                {
                        res = optimize::sgd(problem, x, m_epochs * samples.size(), m_epsilon, updater);
                }
                else
                {
                        log_error() << "stochastic trainer: invalid optimization method <" << m_optimizer << ">!";
                        return false;
                }

                model.load_params(res.optimum().x);

                // OK
                log_info() << "stochastic trainer: optimum [loss = " << res.optimum().f
                           << ", gradient = " << res.optimum().g.norm()
                           << ", calls = " << res.n_fval_calls() << "/" << res.n_grad_calls()
                           << "]s, updates = [" << res.iterations()
                           << "], speed = [" << res.speed().avg() << " +/- " << res.speed().stdev() << "].";

                return true;
        }

        //-------------------------------------------------------------------------------------------------
}
