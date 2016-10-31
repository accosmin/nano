#include "loop.h"
#include "model.h"
#include "stochastic.h"
#include "math/numeric.h"
#include "text/to_params.h"
#include "stoch_optimizer.h"
#include "text/from_params.h"

namespace nano
{
        stochastic_trainer_t::stochastic_trainer_t(const string_t& parameters) :
                trainer_t(concat_params(parameters,
                "opt=sg[...],epochs=16[1,1024],policy=stop_early[,all_epochs],min_batch=32[32,1024],max_batch=256[32,4096]"))
        {
        }

        rtrainer_t stochastic_trainer_t::clone() const
        {
                return std::make_unique<stochastic_trainer_t>(*this);
        }

        trainer_result_t stochastic_trainer_t::train(
                const task_t& task, const size_t fold, const size_t nthreads,
                const loss_t& loss, const criterion_t& criterion,
                model_t& model) const
        {
                if (model != task)
                {
                        throw std::runtime_error("stochastic trainer: mis-matching model and task");
                }

                // parameters
                const auto epochs = clamp(from_params<size_t>(config(), "epochs"), 1, 1024);
                const auto policy = from_params<trainer_policy>(config(), "policy");
                const auto batch0 = clamp(from_params<size_t>(config(), "min_batch"), 32, 1024);
                const auto batchK = clamp(from_params<size_t>(config(), "max_batch"), 32, 1024);
                const auto optimizer = from_params<string_t>(config(), "opt");

                const auto train_fold = fold_t{fold, protocol::train};
                const auto train_size = task.n_samples(train_fold);
                const auto samples = epochs * train_size;

                const auto factor = clamp(scalar_t(samples - batch0) / scalar_t(samples - batchK), scalar_t(1), scalar_t(2));
                const auto epoch_size = idiv(static_cast<size_t>(std::log(batchK / batch0) / std::log(factor)), epochs);

                // train the model
                const timer_t timer;
                const auto op = [&] (const accumulator_t& lacc, const accumulator_t& gacc, const vector_t& x0)
                {
                        size_t epoch = 0;
                        trainer_result_t result;

                        task_iterator_t it(task, train_fold, batch0, factor);

                        // tuning operator
                        const auto fn_tlog = [&] (const state_t& state, const trainer_config_t& sconfig)
                        {
                                lacc.set_params(state.x);
                                lacc.update(task, train_fold);
                                const auto train = trainer_measurement_t{lacc.value(), lacc.vstats(), lacc.estats()};

                                const auto config = nano::append(sconfig, "lambda", lacc.lambda());

                                log_info()
                                        << "[tune:train=" << train
                                        << "," << config << ",batch=" << (it.end() - it.begin())
                                        << "] " << timer.elapsed() << ".";

                                // NB: need to reset the minibatch size (changed during tuning)!
                                it.reset(batch0, factor);

                                return train.m_value;
                        };

                        // logging operator
                        const auto fn_ulog = [&] (const state_t& state, const trainer_config_t& sconfig)
                        {
                                return ulog(lacc, it, epoch, epochs, result, policy, timer, state, sconfig);
                        };

                        // assembly optimization problem & optimize the model
                        const auto problem = make_trainer_problem(lacc, gacc, it);
                        const auto params = stoch_params_t{epochs, epoch_size, fn_ulog, fn_tlog};

                        get_stoch_optimizers().get(optimizer)->minimize(params, problem, x0);

                        return result;
                };

                const auto result = trainer_loop(model, nthreads, loss, criterion, op);
                log_info() << "<<< stoch-" << optimizer << ": " << result << ",time=" << timer.elapsed() << ".";

                // OK
                if (result.valid())
                {
                        model.load_params(result.optimum_params());
                }
                return result;
        }
}
