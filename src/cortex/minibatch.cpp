#include "timer.h"
#include "logger.h"
#include "iterator.h"
#include "minibatch.h"
#include "math/batch.hpp"
#include "thread/thread.h"
#include "trainer_loop.hpp"
#include "text/to_string.hpp"

namespace nano
{
        static trainer_result_t train(
                const task_t& task, const fold_t& tfold, const fold_t& vfold,
                const accumulator_t& lacc, const accumulator_t& gacc,
                const vector_t& x0, const batch_optimizer optimizer, const size_t epochs, const scalar_t epsilon,
                const bool verbose)
        {
                const timer_t timer;

                trainer_result_t result;

                const auto train_size = task.n_samples(tfold);
                const auto batch_size = 32 * nano::n_threads();
                const auto epoch_size = (train_size + batch_size - 1) / batch_size;
                const auto epoch_iterations = size_t(4);
                const auto history_size = epoch_iterations;

                minibatch_iterator_t<shuffle::on> iter(task, tfold, batch_size);

                // construct the optimization problem
                const auto fn_size = [&] ()
                {
                        return lacc.psize();
                };

                const auto fn_fval = [&] (const vector_t& x)
                {
                        lacc.set_params(x);
                        lacc.update(task, tfold, iter.begin(), iter.end());
                        return lacc.value();
                };

                const auto fn_grad = [&] (const vector_t& x, vector_t& gx)
                {
                        gacc.set_params(x);
                        gacc.update(task, tfold, iter.begin(), iter.end());
                        gx = gacc.vgrad();
                        return gacc.value();
                };

                const auto fn_ulog = nullptr;

                // optimize the model
                vector_t x = x0;

                for (size_t epoch = 1; epoch <= epochs; ++ epoch)
                {
                        // optimize mini-batches in sequence
                        for (size_t i = 0; i < epoch_size; ++ i)
                        {
                                const auto state = nano::minimize(
                                        opt_problem_t(fn_size, fn_fval, fn_grad), fn_ulog,
                                        x, optimizer, epoch_iterations, epsilon, history_size);
                                x = state.x;
                                iter.next();
                        }

                        // evaluate training samples
                        lacc.set_params(x);
                        lacc.update(task, tfold);
                        const auto tvalue = lacc.value();
                        const auto terror_avg = lacc.avg_error();
                        const auto terror_var = lacc.var_error();

                        // evaluate validation samples
                        lacc.set_params(x);
                        lacc.update(task, vfold);
                        const auto vvalue = lacc.value();
                        const auto verror_avg = lacc.avg_error();
                        const auto verror_var = lacc.var_error();

                        // OK, update the optimum state
                        const auto milis = timer.milliseconds();
                        const auto ret = result.update(x,
                                {milis, epoch, tvalue, terror_avg, terror_var, vvalue, verror_avg, verror_var},
                                {{"lambda", lacc.lambda()}});

                        if (verbose)
                        log_info()
                                << "[train = " << tvalue << "/" << terror_avg
                                << ", valid = " << vvalue << "/" << verror_avg
                                << " (" << nano::to_string(ret) << ")"
                                << ", epoch = " << epoch << "/" << epochs
                                << ", batch = " << batch_size
                                << ", lambda = " << lacc.lambda()
                                << "] done in " << timer.elapsed() << ".";

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
                const auto op = [&] (
                        const auto& tfold, const auto& vfold, const auto& lacc, const auto& gacc, const auto& x0)
                {
                        return train(task, tfold, vfold, lacc, gacc, x0, optimizer, epochs, epsilon, verbose);
                };

                return trainer_loop(model, task, fold, nthreads, loss, criterion, op);
        }
}

