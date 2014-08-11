#include "trainer.h"
#include "common/timer.h"
#include "common/logger.h"
#include "common/thread_pool.h"
#include "common/random.hpp"
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

        trainer_result_t::trainer_result_t(size_t n_parameters, size_t epochs)
                :       m_opt_params(n_parameters),
                        m_opt_epoch(0),
                        m_epochs(epochs)
        {
        }

        bool trainer_result_t::update(const vector_t& params,
                    scalar_t tvalue, scalar_t terror,
                    scalar_t vvalue, scalar_t verror,
                    size_t epoch, const scalars_t& config)
        {
                const trainer_state_t state(tvalue, terror, vvalue, verror);
                m_history[config].push_back(state);
                
                if (verror < m_opt_state.m_verror)
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

        namespace detail
        {        
                static opt_state_t batch_train(
                        const task_t& task, const sampler_t& tsampler, const sampler_t& vsampler,
                        const loss_t& loss, batch_optimizer optimizer, size_t epochs, size_t iterations, scalar_t epsilon,
                        const vector_t& x0, 
                        accumulator_t& ldata, accumulator_t& gdata, 
                        trainer_result_t& result, size_t& epoch)
                {
                        size_t iteration = 0;  
                        
                        samples_t tsamples = tsampler.get();
                        samples_t vsamples = vsampler.get();                        

                        // construct the optimization problem
                        const timer_t timer;

                        auto fn_size = [&] ()
                        {
                                return ldata.psize();
                        };

                        auto fn_fval = [&] (const vector_t& x)
                        {
                                // training samples: loss value
                                ldata.reset(x);
                                ldata.update(task, tsamples, loss);
                                const scalar_t tvalue = ldata.value();

                                return tvalue;
                        };

                        auto fn_fval_grad = [&] (const vector_t& x, vector_t& gx)
                        {
                                // training samples: loss value & gradient
                                gdata.reset(x);
                                gdata.update(task, tsamples, loss);
                                const scalar_t tvalue = gdata.value();
                                gx = gdata.vgrad();

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
                                        
                                        const scalar_t tvalue = gdata.value();
                                        const scalar_t terror = gdata.error();

                                        // validation samples: loss value
                                        ldata.reset(state.x);
                                        ldata.update(task, vsamples, loss);
                                        const scalar_t vvalue = ldata.value();
                                        const scalar_t verror = ldata.error();

                                        // update the optimum state
                                        result.update(state.x, tvalue, terror, vvalue, verror, epoch,
                                                      scalars_t({ ldata.lambda() }));

                                        log_info() << "[train = " << tvalue << "/" << terror
                                                << ", valid = " << vvalue << "/" << verror
                                                << ", grad = " << state.g.lpNorm<Eigen::Infinity>()
                                                << ", dims = " << ldata.psize()
                                                << ", lambda = " << ldata.lambda()
                                                << "] done in " << timer.elapsed() << ".";
                                                
                                        // resample
                                        if (iteration < iterations)
                                        {
                                                tsamples = tsampler.get();
                                                vsamples = vsampler.get();  
                                        }
                                }
                        };

                        // assembly optimization problem & optimize the model
                        const opt_problem_t problem(fn_size, fn_fval, fn_fval_grad);

                        const opt_opulog_t fn_ulog_ref = std::bind(fn_ulog, _1, std::ref(timer));

                        switch (optimizer)
                        {
                        case batch_optimizer::LBFGS:
                                return optimize::lbfgs(problem, x0, epochs * iterations, epsilon,
                                                       fn_wlog, fn_elog, fn_ulog_ref);

                        case batch_optimizer::CGD:
                                return optimize::cgd_pr(problem, x0, epochs * iterations, epsilon,
                                                        fn_wlog, fn_elog, fn_ulog_ref);

                        case batch_optimizer::GD:
                        default:
                                return optimize::gd(problem, x0, epochs * iterations, epsilon,
                                                    fn_wlog, fn_elog, fn_ulog_ref);
                        }
                }
        }
        
        bool batch_train(
                const task_t& task, const sampler_t& tsampler, const sampler_t& vsampler, size_t nthreads,
                const loss_t& loss, const string_t& criterion, 
                batch_optimizer optimizer, size_t cycles, size_t epochs, size_t iterations, scalar_t epsilon,
                const model_t& model, trainer_result_t& result)
        {
                const vector_t x0 = model.params();
                
                // tune the regularization factor (if needed)
                const scalars_t lambdas = accumulator_t::can_regularize(criterion) ? 
                        scalars_t{ 0.1, 0.2, 0.5, 1.0 } : scalars_t{ 0.0 };                
                for (scalar_t lambda : lambdas)
                {                        
                        accumulator_t ldata(model, nthreads, criterion, criterion_t::type::value, lambda);
                        accumulator_t gdata(model, nthreads, criterion, criterion_t::type::vgrad, lambda);                        
                        
                        vector_t x = x0;
                        for (size_t c = 0, epoch = 0; c < cycles; c ++)
                        {                        
                                const opt_state_t state = detail::batch_train(
                                        task, tsampler, vsampler,  
                                        loss, optimizer, epochs, iterations, epsilon,
                                        x, ldata, gdata, result, epoch);
                                x = state.x;
                        }
                }
                
                // OK
                return true;
        }

        namespace detail
        {
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

                static void stochastic_train(
                        const task_t& task, const samples_t& tsamples_, const samples_t& vsamples, const loss_t& loss,
                        stochastic_optimizer type, size_t epochs, scalar_t alpha0, scalar_t beta,
                        const vector_t& x0, accumulator_t& ldata, accumulator_t& gdata, trainer_result_t& result,
                        thread_pool_t::mutex_t& mutex)
                {
                        samples_t tsamples = tsamples_;

                        random_t<size_t> xrng(0, tsamples.size());
                        rnd_t xrnd(xrng);

                        timer_t timer;

                        vector_t x = x0, xparam = x, xavg = x;

                        vector_t gavg(x.size());
                        gavg.setZero();

                        scalar_t alpha = alpha0;
                        scalar_t sumb = 1.0 / alpha;

                        for (size_t e = 0; e < epochs; e ++)
                        {
                                std::random_shuffle(tsamples.begin(), tsamples.end(), xrnd);

                                // one epoch: a pass through all training samples
                                switch (type)
                                {
                                        // stochastic gradient
                                case stochastic_optimizer::SG:
                                        for (size_t i = 0; i < tsamples.size(); i ++, alpha *= beta)
                                        {
                                                gdata.reset(x);
                                                gdata.update(task, tsamples[i], loss);

                                                x.noalias() -= alpha * gdata.vgrad();
                                        }
                                        xparam = x;
                                        break;

                                        // stochastic gradient average
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
                                        xparam = x;
                                        break;

                                        // stochastic iterative average
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
                                        xparam = xavg;
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

                                result.update(xparam, tvalue, terror, vvalue, verror, e, 
                                              scalars_t({ alpha0, ldata.lambda() }));

                                log_info()
                                        << "[train = " << tvalue << "/" << terror
                                        << ", valid = " << vvalue << "/" << verror
                                        << ", rate = " << alpha << "/" << alpha0
                                        << ", epoch = " << e << "/" << epochs
                                        << ", dims = " << ldata.psize()
                                        << ", lambda = " << ldata.lambda()
                                        << "] done in " << timer.elapsed() << ".";
                        }
                }
        }

        bool stochastic_train(
                const task_t& task, const sampler_t& tsampler, const sampler_t& vsampler, size_t nthreads,
                const loss_t& loss, const string_t& criterion,
                stochastic_optimizer optimizer, size_t epochs,
                const model_t& model, trainer_result_t& result)
        {                
                // prepare workers
                thread_pool_t wpool(nthreads);
                thread_pool_t::mutex_t mutex;

                const size_t iterations = epochs * tsampler.size();             // SGD iterations
                const scalar_t beta = std::pow(0.01, 1.0 / iterations);         // Learning rate decay rate
                
                const vector_t x0 = model.params();
                
                // tune the regularization factor (if needed)
                const scalars_t lambdas = accumulator_t::can_regularize(criterion) ? 
                        scalars_t{ 0.1, 0.2, 0.5, 1.0 } : scalars_t{ 0.0 };
                for (scalar_t lambda : lambdas)
                {       
                        // tune the learning rate
                        const scalars_t alphas = { 0.001, 0.010, 0.100 };
                        for (scalar_t alpha : alphas)
                        {
                                wpool.enqueue([=, &task, &loss, &model, &x0, &result, &mutex]()
                                {
                                        accumulator_t ldata(model, 1, criterion, criterion_t::type::value, lambda);
                                        accumulator_t gdata(model, 1, criterion, criterion_t::type::vgrad, lambda);                                
                                        
                                        const samples_t tsamples = tsampler.get();
                                        const samples_t vsamples = vsampler.get();
                                        
                                        detail::stochastic_train(
                                                task, tsamples, vsamples, loss,
                                                optimizer, epochs, alpha, beta,
                                                x0, ldata, gdata, result, mutex);
                                });
                        }
                }

                wpool.wait();

                // OK
                return true;
        }
}
	
