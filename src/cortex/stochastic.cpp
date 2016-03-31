#include "timer.h"
#include "logger.h"
#include "iterator.h"
#include "stochastic.h"
#include "math/stoch.hpp"
#include "thread/thread.h"
#include "trainer_loop.hpp"
#include "text/to_string.hpp"

namespace nano
{
        static trainer_result_t train(
                const task_t& task, const fold_t& tfold, const fold_t& vfold,
                const accumulator_t& lacc, const accumulator_t& gacc,
                const vector_t& x0, const stoch_optimizer optimizer, const size_t epochs,
                const bool verbose)
        {
                const nano::timer_t timer;

                trainer_result_t result;

                const auto train_size = task.n_samples(tfold);
                const auto batch_size = 16 * nano::n_threads();
                const auto epoch_size = (train_size + batch_size - 1) / batch_size;

                size_t epoch = 0;

                minibatch_iterator_t<shuffle::on> iter(task, tfold, batch_size);

                // construct the optimization problem
                const auto fn_size = [&] ()
                {
                        return lacc.psize();
                };

                const auto fn_fval = [&] (const vector_t& x)
                {
                        iter.next();
                        lacc.set_params(x);
                        lacc.update(task, tfold, iter.begin(), iter.end());
                        return lacc.value();
                };

                const auto fn_grad = [&] (const vector_t& x, vector_t& gx)
                {
                        iter.next();
                        gacc.set_params(x);
                        gacc.update(task, tfold, iter.begin(), iter.end());
                        gx = gacc.vgrad();
                        return gacc.value();
                };

                auto fn_tlog = [&] (const opt_state_t& state, const auto& config)
                {
                        // evaluate training samples
                        lacc.set_params(state.x);
                        lacc.update(task, tfold);
                        const auto tvalue = lacc.value();
                        const auto terror_avg = lacc.avg_error();

                        if (verbose)
                        log_info()
                                << "[tune = " << tvalue << "/" << terror_avg
                                << ", batch = " << batch_size
                                << ", " << append(config, "lambda", lacc.lambda())
                                << "] done in " << timer.elapsed() << ".";

                        return tvalue;
                };

                auto fn_ulog = [&] (const opt_state_t& state, const auto& config)
                {
                        // evaluate training samples
                        lacc.set_params(state.x);
                        lacc.update(task, tfold);
                        const auto tvalue = lacc.value();
                        const auto terror_avg = lacc.avg_error();
                        const auto terror_var = lacc.var_error();

                        // evaluate validation samples
                        lacc.set_params(state.x);
                        lacc.update(task, vfold);
                        const auto vvalue = lacc.value();
                        const auto verror_avg = lacc.avg_error();
                        const auto verror_var = lacc.var_error();

                        // OK, update the optimum solution
                        const auto milis = timer.milliseconds();
                        const auto ret = result.update(state.x,
                                {milis, ++ epoch, tvalue, terror_avg, terror_var, vvalue, verror_avg, verror_var},
                                nano::append(config, "lambda", lacc.lambda()));

                        if (verbose)
                        log_info()
                                << "[train = " << tvalue << "/" << terror_avg
                                << ", valid = " << vvalue << "/" << verror_avg
                                << " (" << nano::to_string(ret) << ")"
                                << ", epoch = " << epoch << "/" << epochs
                                << ", batch = " << batch_size
                                << ", " << append(config, "lambda", lacc.lambda())
                                << "] done in " << timer.elapsed() << ".";

                        return !nano::is_done(ret);
                };

                // optimize the model
                nano::minimize(
                        opt_problem_t(fn_size, fn_fval, fn_grad), fn_ulog, fn_tlog,
                        x0, optimizer, epochs, epoch_size);

                return result;
        }

        trainer_result_t stochastic_train(
                const model_t& model, const task_t& task, const size_t fold, const size_t nthreads,
                const loss_t& loss, const criterion_t& criterion,
                const stoch_optimizer optimizer, const size_t epochs, const bool verbose)
        {
                const auto op = [&] (
                        const auto& tfold, const auto& vfold, const auto& lacc, const auto& gacc, const auto& x0)
                {
                        return train(task, tfold, vfold, lacc, gacc, x0, optimizer, epochs, verbose);
                };

                return trainer_loop(model, task, fold, nthreads, loss, criterion, op);
        }
}
