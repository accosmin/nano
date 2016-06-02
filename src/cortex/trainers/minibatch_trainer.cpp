#include "math/clamp.hpp"
#include "cortex/model.h"
#include "optim/batch.hpp"
#include "thread/thread.h"
#include "math/numeric.hpp"
#include "text/to_string.hpp"
#include "minibatch_trainer.h"
#include "text/from_params.hpp"
#include "cortex/task_iterator.h"
#include "cortex/trainer_loop.hpp"
#include "math/stoch/stoch_loop.hpp"

#include "cortex/logger.h"

namespace nano
{
        minibatch_trainer_t::minibatch_trainer_t(const string_t& parameters) :
                trainer_t(parameters)
        {
        }

        trainer_result_t minibatch_trainer_t::train(
                const task_t& task, const size_t fold, const size_t nthreads,
                const loss_t& loss, const criterion_t& criterion,
                model_t& model) const
        {
                // initialize the model
                model.resize(task, true);
                model.random_params();

                // parameters
                const auto epochs = clamp(from_params<size_t>(configuration(), "epochs", 16), 1, 1024);
                const auto epsilon = clamp(from_params<scalar_t>(configuration(), "eps", scalar_t(1e-6)), scalar_t(1e-8), scalar_t(1e-3));
                const auto optimizer = from_string<batch_optimizer>(from_params<string_t>(configuration(), "opt", "cgd"));
                const auto verbose = true;

                // train the model
                const auto op = [&] (const auto& lacc, const auto& gacc, const auto& x0)
                {
                        return train(task, fold, lacc, gacc, x0, optimizer, epochs, epsilon, verbose);
                };

                const auto result = trainer_loop(model, nthreads, loss, criterion, op);
                log_info() << "<<< minibatch-" << to_string(optimizer) << ": " << result << ".";

                // OK
                if (result.valid())
                {
                        model.load_params(result.optimum_params());
                }
                return result;
        }

        trainer_result_t minibatch_trainer_t::train(
                const task_t& task, const size_t fold,
                const accumulator_t& lacc, const accumulator_t& gacc, const vector_t& x0,
                const batch_optimizer optimizer, const size_t epochs, const scalar_t epsilon,
                const bool verbose) const
        {
                const timer_t timer;

                const auto train_fold = fold_t{fold, protocol::train};
                const auto valid_fold = fold_t{fold, protocol::valid};
                const auto test_fold = fold_t{fold, protocol::test};

                const auto train_size = task.n_samples(train_fold);
                const auto batch_size = 32 * thread::concurrency();
                const auto epoch_size = idiv(train_size, batch_size);
                const auto epoch_iterations = size_t(4);
                const auto history_size = epoch_iterations;

                size_t epoch = 0;
                trainer_result_t result;

                minibatch_iterator_t<shuffle::on> iter(task, train_fold, batch_size);

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

                const auto fn_ulog = [&] (const opt_state_t& state, const auto& sconfig)
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
                        const auto config = nano::append(sconfig, "lambda", lacc.lambda());
                        const auto ret = result.update(state.x, {milis, ++epoch, train, valid, test}, config);

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

                        return !nano::is_done(ret);
                };

                const auto op = [&] (opt_state_t& state)
                {
                        state = nano::minimize(
                                opt_problem_t(fn_size, fn_fval, fn_grad), nullptr,
                                state.x, optimizer, epoch_iterations, epsilon, history_size);
                        iter.next();
                };

                // assembly optimization problem & optimize the model
                const auto problem = opt_problem_t(fn_size, fn_fval, fn_grad);
                const auto params = stoch_params_t<opt_problem_t>(epochs, epoch_size, fn_ulog);
                const auto config = stoch_params_t<opt_problem_t>::tconfig();

                nano::stoch_loop(problem, params, opt_state_t(problem, x0), op, config);

                return result;
        }
}
