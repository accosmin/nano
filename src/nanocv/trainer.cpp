#include "trainer.h"
#include "common/timer.h"
#include "common/logger.h"
#include "common/thread_pool.h"
#include "common/random.hpp"
#include "common/log_search.hpp"
#include "optimize/opt_gd.hpp"
#include "optimize/opt_cgd.hpp"
#include "optimize/opt_lbfgs.hpp"
#include "accumulator.h"
#include "sampler.h"
#include "loss.h"
#include <fstream>

namespace ncv
{
        bool save(const string_t& path, const trainer_states_t& states)
        {
                std::ofstream ofs(path.c_str(), std::ofstream::out);
                if (!ofs.is_open())
                {
                        return false;
                }
                
                const string_t delim = "\t";              
                
                // header
                ofs 
                << text::resize("train-loss", 16) << delim
                << text::resize("train-error", 16) << delim
                << text::resize("valid-loss", 16) << delim
                << text::resize("valid-error", 16) << delim << "\n";

                // optimization states
                for (const trainer_state_t& state : states)
                {
                        ofs 
                        << text::resize(text::to_string(state.m_tvalue), 16) << delim
                        << text::resize(text::to_string(state.m_terror), 16) << delim
                        << text::resize(text::to_string(state.m_vvalue), 16) << delim
                        << text::resize(text::to_string(state.m_verror), 16) << delim << "\n";
                }

                return ofs.good();
        }

        trainer_result_t::trainer_result_t()
                :       m_opt_epoch(0)
        {
        }

        bool trainer_result_t::update(const vector_t& params,
                    scalar_t tvalue, scalar_t terror,
                    scalar_t vvalue, scalar_t verror,
                    size_t epoch, const scalars_t& config)
        {
                const trainer_state_t state(tvalue, terror, vvalue, verror);
                m_history[config].push_back(state);
                
                if (state < m_opt_state)
                {
                        m_opt_params = params;
                        m_opt_state = state;
                        m_opt_epoch = epoch;
                        m_opt_config = config;

                        return true;
                }

                else
                {
                        return false;
                }
        }

        bool trainer_result_t::update(const trainer_result_t& other)
        {
                if (*this < other)
                {
                        *this = other;
                        return true;
                }

                else
                {
                        return false;
                }
        }
        
        trainer_states_t trainer_result_t::optimum_states() const
        {
                const string_t str_opt_config = text::concatenate(m_opt_config, "-");
                for (const auto& it : m_history)
                {
                        const string_t str_config = text::concatenate(it.first, "-");
                        if (str_config == str_opt_config)
                        {
                                return it.second;
                        }
                }
                
                return trainer_states_t();
        }
        
        trainer_data_t::trainer_data_t(const task_t& task,
                       const sampler_t& tsampler,
                       const sampler_t& vsampler,
                       const loss_t& loss,
                       const vector_t& x0,
                       accumulator_t& lacc,
                       accumulator_t& gacc)
                :       m_task(task),
                        m_tsampler(tsampler),
                        m_vsampler(vsampler),
                        m_loss(loss),
                        m_x0(x0),
                        m_lacc(lacc),
                        m_gacc(gacc)
        {
        }

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
                                return log_min_search<toperator, scalar_t>(op, -1.0, +6.0, 0.2, 4);
                        }

                        else
                        {
                                return op(0.0);
                        }
                }
        }

        namespace detail
        {        
                static opt_state_t batch_train(
                        trainer_data_t& data, 
                        batch_optimizer optimizer, size_t epochs, size_t iterations, scalar_t epsilon, size_t& epoch,
                        trainer_result_t& result)
                {
                        size_t iteration = 0;  
                        
                        samples_t tsamples = data.m_tsampler.get();
                        samples_t vsamples = data.m_vsampler.get();

                        // construct the optimization problem
                        const timer_t timer;

                        auto fn_size = [&] ()
                        {
                                return data.m_lacc.psize();
                        };

                        auto fn_fval = [&] (const vector_t& x)
                        {
                                // training samples: loss value
                                data.m_lacc.reset(x);
                                data.m_lacc.update(data.m_task, tsamples, data.m_loss);
                                const scalar_t tvalue = data.m_lacc.value();

                                return tvalue;
                        };

                        auto fn_fval_grad = [&] (const vector_t& x, vector_t& gx)
                        {
                                // training samples: loss value & gradient
                                data.m_gacc.reset(x);
                                data.m_gacc.update(data.m_task, tsamples, data.m_loss);
                                const scalar_t tvalue = data.m_gacc.value();
                                gx = data.m_gacc.vgrad();

                                return tvalue;
                        };

                        auto fn_wlog = [] (const string_t& message)
                        {
                                log_warning() << message;
                        };
                        auto fn_elog = [] (const string_t& message)
                        {
                                log_error() << message;
                        };
                        auto fn_ulog = [&] (const opt_state_t& state, const timer_t& timer)
                        {
                                ++ iteration;
                                if ((iteration % iterations) == 0)
                                {
                                        ++ epoch;
                                        
                                        const scalar_t tvalue = data.m_gacc.value();
                                        const scalar_t terror = data.m_gacc.error();

                                        // validation samples: loss value
                                        data.m_lacc.reset(state.x);
                                        data.m_lacc.update(data.m_task, vsamples, data.m_loss);
                                        const scalar_t vvalue = data.m_lacc.value();
                                        const scalar_t verror = data.m_lacc.error();

                                        // update the optimum state
                                        result.update(state.x, tvalue, terror, vvalue, verror, epoch,
                                                      scalars_t({ data.m_lacc.lambda() }));

                                        log_info()
                                                << "[train = " << tvalue << "/" << terror
                                                << ", valid = " << vvalue << "/" << verror
                                                << ", grad = " << state.g.lpNorm<Eigen::Infinity>()
                                                << ", eval = " << state.n_fval_calls() << "/" << state.n_grad_calls()
                                                << ", dims = " << data.m_lacc.psize()
                                                << ", lambda = " << data.m_lacc.lambda()
                                                << "] done in " << timer.elapsed() << ".";
                                                
                                        // resample
                                        if (iteration < iterations)
                                        {
                                                tsamples = data.m_tsampler.get();
                                                vsamples = data.m_vsampler.get();  
                                        }
                                }
                        };

                        // assembly optimization problem & optimize the model
                        const opt_problem_t problem(fn_size, fn_fval, fn_fval_grad);

                        const opt_opulog_t fn_ulog_ref = std::bind(fn_ulog, _1, std::ref(timer));

                        switch (optimizer)
                        {
                        case batch_optimizer::LBFGS:
                                return optimize::lbfgs(problem, data.m_x0, epochs * iterations, epsilon,
                                                       fn_wlog, fn_elog, fn_ulog_ref);

                        case batch_optimizer::CGD:
                                return optimize::cgd_pr(problem, data.m_x0, epochs * iterations, epsilon,
                                                        fn_wlog, fn_elog, fn_ulog_ref);

                        case batch_optimizer::GD:
                        default:
                                return optimize::gd(problem, data.m_x0, epochs * iterations, epsilon,
                                                    fn_wlog, fn_elog, fn_ulog_ref);
                        }
                }
        }
        
        trainer_result_t batch_train(
                const model_t& model, const task_t& task, const sampler_t& tsampler, const sampler_t& vsampler, size_t nthreads,
                const loss_t& loss, const string_t& criterion, 
                batch_optimizer optimizer, size_t cycles, size_t epochs, size_t iterations, scalar_t epsilon)
        {
                const auto op = [&] (scalar_t lambda)
                {
                        accumulator_t lacc(model, nthreads, criterion, criterion_t::type::value, lambda);
                        accumulator_t gacc(model, nthreads, criterion, criterion_t::type::vgrad, lambda);

                        trainer_result_t result;

                        // optimize the model
                        vector_t x;
                        model.save_params(x);
                        for (size_t c = 0, epoch = 0; c < cycles; c ++)
                        {
                                trainer_data_t data(task, tsampler, vsampler, loss, x, lacc, gacc);

                                const opt_state_t state = detail::batch_train(
                                        data, optimizer, epochs, iterations, epsilon, epoch, result);

                                x = state.x;
                        }

                        // OK
                        return result;
                };

                // tune the regularization factor (if needed)
                return detail::tune(op, criterion);
        }
}
	
