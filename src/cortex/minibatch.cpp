#include "timer.h"
#include "minibatch.h"
#include "math/batch.hpp"
#include "task_iterator.h"
#include "thread/thread.h"
#include "trainer_loop.hpp"
#include "math/momentum.hpp"
#include "text/to_string.hpp"
#include "logger.h"

namespace nano
{
        static trainer_result_t train(
                const task_t& task, const size_t fold,
                const accumulator_t& lacc, const accumulator_t& gacc,
                const vector_t& x0, const batch_optimizer optimizer, const size_t epochs, const scalar_t epsilon,
                const bool verbose)
        {
                const timer_t timer;

                const auto train_fold = fold_t{fold, protocol::train};
                const auto valid_fold = fold_t{fold, protocol::valid};
                const auto test_fold = fold_t{fold, protocol::test};

                const auto train_size = task.n_samples(train_fold);
                const auto batch_size = 32 * thread::concurrency();
                const auto epoch_size = (train_size + batch_size - 1) / batch_size;
                const auto epoch_iterations = size_t(4);
                const auto history_size = epoch_iterations;

                minibatch_iterator_t<shuffle::on> iter(task, train_fold, batch_size);

                trainer_result_t result;

                // construct the optimization problem
                const auto fn_size = [&] ()
                {
                        return lacc.psize();
                };

                const auto fn_fval = [&] (const vector_t& x)
                {
                        lacc.set_params(x);
                        lacc.update(task, iter.fold(), iter.begin(), iter.end());
                        return lacc.value();
                };

                const auto fn_grad = [&] (const vector_t& x, vector_t& gx)
                {
                        gacc.set_params(x);
                        gacc.update(task, iter.fold(), iter.begin(), iter.end());
                        gx = gacc.vgrad();
                        return gacc.value();
                };

                const auto fn_ulog = nullptr;

                // optimize the model
                vector_t x = x0;

                const scalar_t momentum = 0.95;
                momentum_vector_t<vector_t> xavg(momentum, x0.size());

                for (size_t epoch = 1; epoch <= epochs; ++ epoch)
                {
                        // optimize mini-batches in sequence
                        for (size_t i = 0; i < epoch_size; ++ i)
                        {
                                const auto state = nano::minimize(
                                        opt_problem_t(fn_size, fn_fval, fn_grad), fn_ulog,
                                        x, optimizer, epoch_iterations, epsilon, history_size);
                                xavg.update(state.x);
                                x = state.x;
                                iter.next();
                        }

                        // evaluate the current state
                        lacc.set_params(xavg.value());

                        lacc.update(task, train_fold);
                        const auto train = trainer_measurement_t{lacc.value(), lacc.avg_error(), lacc.var_error()};

                        lacc.update(task, valid_fold);
                        const auto valid = trainer_measurement_t{lacc.value(), lacc.avg_error(), lacc.var_error()};

                        lacc.update(task, test_fold);
                        const auto test = trainer_measurement_t{lacc.value(), lacc.avg_error(), lacc.var_error()};

                        // OK, update the optimum state
                        const auto milis = timer.milliseconds();
                        const auto config = trainer_config_t{{"lambda", lacc.lambda()}};
                        const auto ret = result.update(xavg.value(), {milis, epoch, train, valid, test}, config);

                        if (verbose)
                        {
                                log_info()
                                        << "[" << epoch << "/" << epochs
                                        << ": train=" << train
                                        << ", valid=" << valid << "|" << nano::to_string(ret)
                                        << ", test=" << test
                                        << ", " << config << ",batch=" << batch_size
                                        << "] " << timer.elapsed() << ".";
                        }

                        if (nano::is_done(ret))
                        {
                                break;
                        }
                }

                return result;
        }

        trainer_result_t minibatch_train(
                const model_t& model, const task_t& task, const size_t fold, const size_t nthreads,
                const loss_t& loss, const criterion_t& criterion,
                const batch_optimizer optimizer, const size_t epochs, const scalar_t epsilon, const bool verbose)
        {
                const auto op = [&] (const auto& lacc, const auto& gacc, const auto& x0)
                {
                        return train(task, fold, lacc, gacc, x0, optimizer, epochs, epsilon, verbose);
                };

                return trainer_loop(model, nthreads, loss, criterion, op);
        }
}

