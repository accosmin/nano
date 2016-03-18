#include "task.h"
#include "batch.h"
#include "timer.h"
#include "logger.h"
#include "accumulator.h"
#include "math/tune.hpp"
#include "math/batch.hpp"
#include "text/to_string.hpp"

namespace nano
{
        static trainer_result_t train(
                const task_t& task, const fold_t& tfold, const fold_t& vfold,
                const accumulator_t& lacc, const accumulator_t& gacc,
                const vector_t& x0, const batch_optimizer optimizer, const size_t iterations, const scalar_t epsilon,
                const bool verbose)
        {
                const timer_t timer;

                trainer_result_t result;

                size_t iteration = 0;

                // construct the optimization problem
                const auto fn_size = [&] ()
                {
                        return lacc.psize();
                };

                const auto fn_fval = [&] (const vector_t& x)
                {
                        lacc.set_params(x);
                        lacc.update(task, tfold);
                        return lacc.value();
                };

                const auto fn_grad = [&] (const vector_t& x, vector_t& gx)
                {
                        gacc.set_params(x);
                        gacc.update(task, tfold);
                        gx = gacc.vgrad();
                        return gacc.value();
                };

                auto fn_ulog = [&] (const opt_state_t& state)
                {
                        // evaluate training samples
                        const auto tvalue = gacc.value();
                        const auto terror_avg = gacc.avg_error();
                        const auto terror_var = gacc.var_error();

                        // evaluate validation samples
                        lacc.set_params(state.x);
                        lacc.update(task, vfold);
                        const scalar_t vvalue = lacc.value();
                        const scalar_t verror_avg = lacc.avg_error();
                        const scalar_t verror_var = lacc.var_error();

                        // OK, update the optimum state
                        const auto milis = timer.milliseconds();
                        const auto ret = result.update(state.x,
                                {milis, ++ iteration, tvalue, terror_avg, terror_var, vvalue, verror_avg, verror_var},
                                {{"lambda", lacc.lambda()}});

                        if (verbose)
                        log_info()
                                << "[train = " << tvalue << "/" << terror_avg
                                << ", valid = " << vvalue << "/" << verror_avg
                                << " (" << nano::to_string(ret) << ")"
                                << ", epoch = " << iteration << "/" << iterations
                                << ", lambda = " << lacc.lambda()
                                << ", calls = " << state.m_fcalls << "/" << state.m_gcalls
                                << "] done in " << timer.elapsed() << ".";

                        return !nano::is_done(ret);
                };

                // assembly optimization problem & optimize the model
                nano::minimize(
                        opt_problem_t(fn_size, fn_fval, fn_grad), fn_ulog,
                        x0, optimizer, iterations, epsilon);

                return result;
        }

        trainer_result_t batch_train(
                const model_t& model, const task_t& task, const fold_t& tfold, const fold_t& vfold, const size_t nthreads,
                const loss_t& loss, const criterion_t& criterion,
                const batch_optimizer optimizer, const size_t iterations, const scalar_t epsilon,
                const bool verbose)
        {
                vector_t x0;
                model.save_params(x0);

                // setup acumulators
                accumulator_t lacc(model, loss, criterion, criterion_t::type::value); lacc.set_threads(nthreads);
                accumulator_t gacc(model, loss, criterion, criterion_t::type::vgrad); gacc.set_threads(nthreads);

                // tune the regularization factor (if needed)
                const auto op = [&] (scalar_t lambda)
                {
                        lacc.set_lambda(lambda);
                        gacc.set_lambda(lambda);
                        return train(task, tfold, vfold, lacc, gacc, x0, optimizer, iterations, epsilon, verbose);
                };

                if (lacc.can_regularize())
                {
                        const auto space = nano::make_log10_space(-6.0, +6.0, 0.5);
                        return nano::tune(op, space).optimum();
                }
                else
                {
                        return op(0.0);
                }
        }
}

