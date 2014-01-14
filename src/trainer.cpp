#include "trainer.h"
#include "trainer_data.h"
#include "text.h"
#include "util/timer.h"
#include "util/logger.h"
#include "optimize/opt_gd.hpp"
#include "optimize/opt_cgd.hpp"
#include "optimize/opt_lbfgs.hpp"

namespace ncv
{
        /////////////////////////////////////////////////////////////////////////////////////////

        bool train(
                const task_t& task, const samples_t& tsamples, const samples_t& vsamples, const loss_t& loss,
                const string_t& optimizer, scalar_t epsilon, size_t iterations, size_t nthreads,
                model_t& model, trainer_state_t& state, bool verbose)
        {
                // construct the optimization problem
                timer_t timer;

                trainer_data_skipgrad_t ldata(model);
                trainer_data_withgrad_t gdata(model);

                auto fn_size = [&] ()
                {
                        return ldata.n_parameters();
                };

                auto fn_fval = [&] (const vector_t& x)
                {
                        // training samples: loss value
                        ldata.load_params(x);
                        ldata.update_mt(task, tsamples, loss, nthreads);
                        const scalar_t tvalue = ldata.value();
                        const scalar_t terror = ldata.error();

                        // validation samples: loss value
                        ldata.load_params(x);
                        ldata.update_mt(task, vsamples, loss, nthreads);
                        const scalar_t vvalue = ldata.value();
                        const scalar_t verror = ldata.error();

                        // update the optimum state
                        state.update(x, tvalue, terror, vvalue, verror);

                        return tvalue;
                };

                auto fn_fval_grad = [&] (const vector_t& x, vector_t& gx)
                {
                        // training samples: loss value & gradient
                        gdata.load_params(x);
                        gdata.update_mt(task, tsamples, loss, nthreads);
                        const scalar_t tvalue = gdata.value();
                        const scalar_t terror = gdata.error();
                        gx = gdata.vgrad();

                        // validation samples: loss value
                        ldata.load_params(x);
                        ldata.update_mt(task, vsamples, loss, nthreads);
                        const scalar_t vvalue = ldata.value();
                        const scalar_t verror = ldata.error();

                        // update the optimum state
                        state.update(x, tvalue, terror, vvalue, verror);

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
                auto fn_ulog = [&] (const opt_state_t& result, timer_t& timer)
                {
                        if (verbose)
                        {
                                const scalar_t tvalue = state.m_tvalue;
                                const scalar_t terror = state.m_terror;
                                const scalar_t vvalue = state.m_vvalue;
                                const scalar_t verror = state.m_verror;

                                log_info() << "[loss = " << result.f
                                           << ", grad = " << result.g.lpNorm<Eigen::Infinity>()
                                           << ", funs = " << result.n_fval_calls() << "/" << result.n_grad_calls()
                                           << ", train* = " << tvalue << "/" << terror
                                           << ", valid* = " << vvalue << "/" << verror
                                           << "] done in " << timer.elapsed() << ".";
                                timer.start();
                        }
                };

                // assembly optimization problem & optimize the model
                const opt_problem_t problem(fn_size, fn_fval, fn_fval_grad);

                const vector_t x = model.params();

                const opt_opulog_t fn_ulog_ref = std::bind(fn_ulog, _1, std::ref(timer));

                if (text::iequals(optimizer, "lbfgs"))
                {
                        optimize::lbfgs(problem, x, iterations, epsilon, fn_wlog, fn_elog, fn_ulog_ref);
                }
                else if (text::iequals(optimizer, "cgd"))
                {
                        optimize::cgd(problem, x, iterations, epsilon, fn_wlog, fn_elog, fn_ulog_ref);
                }
                else if (text::iequals(optimizer, "gd"))
                {
                        optimize::gd(problem, x, iterations, epsilon, fn_wlog, fn_elog, fn_ulog_ref);
                }
                else
                {
                        log_error() << "invalid optimization method <" << optimizer << ">!";
                        return false;
                }

                model.load_params(state.m_params);

                // OK
                return true;
        }

        /////////////////////////////////////////////////////////////////////////////////////////
}
	
