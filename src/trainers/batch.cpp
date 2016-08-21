#include "batch.h"
#include "loop.hpp"
#include "model.h"
#include "logger.h"
#include "math/clamp.hpp"
#include "math/epsilon.hpp"
#include "batch_optimizer.h"
#include "text/to_string.hpp"
#include "text/from_params.hpp"

namespace nano
{
        batch_trainer_t::batch_trainer_t(const string_t& parameters) :
                trainer_t(parameters)
        {
        }

        trainer_result_t batch_trainer_t::train(
                const task_t& task, const size_t fold, const size_t nthreads,
                const loss_t& loss, const criterion_t& criterion,
                model_t& model) const
        {
                if (model != task)
                {
                        throw std::runtime_error("batch trainer: mis-matching model and task");
                }

                // parameters
                const auto epochs = clamp(from_params<size_t>(config(), "epochs"), 4, 4096);
                const auto optimizer = from_params<string_t>(config(), "opt");
                const auto policy = from_params<trainer_policy>(config(), "policy");
                const auto epsilon = epsilon0<scalar_t>();
                const auto verbose = true;

                // train the model
                const auto op = [&] (const accumulator_t& lacc, const accumulator_t& gacc, const vector_t& x0)
                {
                        auto batch_optimizer = get_batch_optimizers().get(optimizer);
                        return train(task, fold, lacc, gacc, x0, *batch_optimizer, epochs, epsilon, policy, verbose);
                };

                const auto result = trainer_loop(model, nthreads, loss, criterion, op);
                log_info() << "<<< batch-" << optimizer << ": " << result << ".";

                // OK
                if (result.valid())
                {
                        model.load_params(result.optimum_params());
                }
                return result;
        }

        trainer_result_t batch_trainer_t::train(
                const task_t& task, const size_t fold,
                const accumulator_t& lacc, const accumulator_t& gacc, const vector_t& x0,
                const batch_optimizer_t& optimizer, const size_t epochs, const scalar_t epsilon,
                const trainer_policy policy, const bool verbose) const
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

                auto fn_ulog = [&] (const state_t& state)
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
                        const auto ret = result.update(state, {milis, ++iteration, train, valid, test}, config);

                        if (verbose)
                        {
                                log_info()
                                        << "[" << iteration << "/" << epochs
                                        << ": train=" << train
                                        << ", valid=" << valid << "|" << nano::to_string(ret)
                                        << ", test=" << test
                                        << ", " << config << ",calls=" << state.m_fcalls << "/" << state.m_gcalls
                                        << "] " << timer.elapsed() << ".";
                        }

                        return !nano::is_done(ret, policy);
                };

                // assembly optimization problem & optimize the model
                optimizer.minimize(
                        batch_params_t(epochs, epsilon, fn_ulog),
                        problem_t(fn_size, fn_fval, fn_grad), x0);

                return result;
        }
}
