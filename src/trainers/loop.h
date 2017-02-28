#pragma once

#include "state.h"
#include "math/tune.h"
#include "accumulator.h"
#include "task_iterator.h"
#include "trainer_result.h"
#include "text/to_params.h"
#include "logger.h"

namespace nano
{
        ///
        /// \brief log the current optimization state & check stopping criteria.
        ///
        inline bool ulog(const accumulator_t& acc, task_iterator_t& it,
                size_t& epoch, const size_t epochs, trainer_result_t& result, const trainer_policy policy, const size_t patience,
                const timer_t& timer,
                const state_t& state, const string_t& sconfig = string_t())
        {
                // evaluate the current state
                // NB: the training state is already estimated!
                const auto train = trainer_measurement_t{acc.value(), acc.vstats(), acc.estats()};

                acc.params(state.x);
                acc.mode(criterion_t::type::value);
                acc.update(it.task(), it.valid_fold());
                const auto valid = trainer_measurement_t{acc.value(), acc.vstats(), acc.estats()};

                acc.params(state.x);
                acc.mode(criterion_t::type::value);
                acc.update(it.task(), it.test_fold());
                const auto test = trainer_measurement_t{acc.value(), acc.vstats(), acc.estats()};

                // OK, update the optimum solution
                const auto milis = timer.milliseconds();
                const auto config = to_params(sconfig, "lambda", acc.lambda());
                const auto xnorm = state.x.lpNorm<2>();
                const auto ret = result.update(state, {milis, ++epoch, xnorm, train, valid, test}, config, patience);

                log_info()
                        << "[" << epoch << "/" << epochs
                        << ":train=" << train
                        << ",valid=" << valid << "|" << nano::to_string(ret)
                        << ",test=" << test
                        << "," << config << ",batch=" << it.size()
                        << ",g=" << state.convergence_criteria() << ",x=" << xnorm
                        << "] " << timer.elapsed() << ".";

                return !nano::is_done(ret, policy);
        }

        ///
        /// \brief generic training loop given a suitable optimization operator.
        ///
        template <typename toperator>      ///< (accumulator, starting_point)
        trainer_result_t trainer_loop(
                const model_t& model, const size_t nthreads,
                const loss_t& loss, const criterion_t& criterion,
                const toperator& trainer)
        {
                vector_t x0;
                model.save_params(x0);

                // setup accumulator
                accumulator_t acc(model, loss, criterion);
                acc.threads(nthreads);

                // tune the regularization factor (if needed)
                const auto op = [&] (const scalar_t lambda)
                {
                        acc.lambda(lambda);
                        return trainer(acc, x0);
                };

                if (acc.can_regularize())
                {
                        const auto space = nano::make_log10_space(scalar_t(-5.0), scalar_t(-1.0), scalar_t(0.5), 4);
                        return nano::tune(op, space).optimum();
                }
                else
                {
                        return op(0);
                }
        }
}
