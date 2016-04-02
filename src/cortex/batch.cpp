#include "task.h"
#include "batch.h"
#include "timer.h"
#include "logger.h"
#include "math/batch.hpp"
#include "trainer_loop.hpp"
#include "text/to_string.hpp"

namespace nano
{
        static trainer_result_t train(
                const task_t& task, const size_t fold,
                const accumulator_t& lacc, const accumulator_t& gacc,
                const vector_t& x0, const batch_optimizer optimizer, const size_t iterations, const scalar_t epsilon,
                const bool verbose)
        {
                const timer_t timer;

                const auto train_fold = fold_t{fold, protocol::train};
                const auto valid_fold = fold_t{fold, protocol::valid};
                const auto test_fold = fold_t{fold, protocol::test};

                size_t iteration = 0;
                trainer_result_t result;

                // construct the optimization problem
                const auto fn_size = [&] ()
                {
                        return lacc.psize();
                };

                const auto fn_fval = [&] (const vector_t& x)
                {
                        lacc.set_params(x);
                        lacc.update(task, train_fold);
                        return lacc.value();
                };

                const auto fn_grad = [&] (const vector_t& x, vector_t& gx)
                {
                        gacc.set_params(x);
                        gacc.update(task, train_fold);
                        gx = gacc.vgrad();
                        return gacc.value();
                };

                auto fn_ulog = [&] (const opt_state_t& state)
                {
                        // evaluate the current state
                        lacc.set_params(state.x);

                        lacc.update(task, train_fold);
                        const auto train = trainer_measurement_t{lacc.value(), lacc.avg_error(), lacc.var_error()};

                        lacc.update(task, valid_fold);
                        const auto valid = trainer_measurement_t{lacc.value(), lacc.avg_error(), lacc.var_error()};

                        lacc.update(task, test_fold);
                        const auto test = trainer_measurement_t{lacc.value(), lacc.avg_error(), lacc.var_error()};

                        // OK, update the optimum solution
                        const auto milis = timer.milliseconds();
                        const auto config = trainer_config_t{{"lambda", lacc.lambda()}};
                        const auto ret = result.update(state.x, {milis, ++iteration, train, valid, test}, config);

                        if (verbose)
                        {
                                log_info()
                                        << "[" << iteration << "/" << iterations
                                        << ": train=" << train
                                        << ", valid=" << valid << "|" << nano::to_string(ret)
                                        << ", test=" << test
                                        << ", " << config << ",calls=" << state.m_fcalls << "/" << state.m_gcalls
                                        << "] " << timer.elapsed() << ".";
                        }

                        return !nano::is_done(ret);
                };

                // assembly optimization problem & optimize the model
                nano::minimize(
                        opt_problem_t(fn_size, fn_fval, fn_grad), fn_ulog,
                        x0, optimizer, iterations, epsilon);

                return result;
        }

        trainer_result_t batch_train(
                const model_t& model, const task_t& task, const size_t fold, const size_t nthreads,
                const loss_t& loss, const criterion_t& criterion,
                const batch_optimizer optimizer, const size_t iterations, const scalar_t epsilon,
                const bool verbose)
        {
                const auto op = [&] (const auto& lacc, const auto& gacc, const auto& x0)
                {
                        return train(task, fold, lacc, gacc, x0, optimizer, iterations, epsilon, verbose);
                };

                return trainer_loop(model, nthreads, loss, criterion, op);
        }
}

