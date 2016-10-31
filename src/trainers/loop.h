#pragma once

#include "state.h"
#include "problem.h"
#include "math/tune.h"
#include "accumulator.h"
#include "task_iterator.h"
#include "trainer_result.h"
#include "text/to_string.h"

#include "logger.h"

namespace nano
{
        ///
        /// \brief construct optimization problem for a particular trainer.
        ///
        inline auto make_trainer_problem(const accumulator_t& lacc, const accumulator_t& gacc, task_iterator_t& it)
        {
                const auto fn_size = [&] ()
                {
                        return lacc.psize();
                };

                const auto fn_fval = [&] (const vector_t& x)
                {
                        it.next();
                        lacc.set_params(x);
                        lacc.update(it.task(), it.fold(), it.begin(), it.end());
                        return lacc.value();
                };

                const auto fn_grad = [&] (const vector_t& x, vector_t& gx)
                {
                        it.next();
                        gacc.set_params(x);
                        gacc.update(it.task(), it.fold(), it.begin(), it.end());
                        gx = gacc.vgrad();
                        return gacc.value();
                };

                return problem_t(fn_size, fn_fval, fn_grad);
        }

        ///
        /// \brief log the current optimization state & check stopping criteria.
        ///
        inline bool ulog(const accumulator_t& lacc, task_iterator_t& it,
                size_t& epoch, const size_t epochs, trainer_result_t& result, const trainer_policy policy,
                const timer_t& timer,
                const state_t& state, const trainer_config_t& sconfig = trainer_config_t())
        {
                // evaluate the current state
                lacc.set_params(state.x);
                lacc.update(it.task(), it.train_fold());
                const auto train = trainer_measurement_t{lacc.value(), lacc.vstats(), lacc.estats()};

                lacc.set_params(state.x);
                lacc.update(it.task(), it.valid_fold());
                const auto valid = trainer_measurement_t{lacc.value(), lacc.vstats(), lacc.estats()};

                lacc.set_params(state.x);
                lacc.update(it.task(), it.test_fold());
                const auto test = trainer_measurement_t{lacc.value(), lacc.vstats(), lacc.estats()};

                // OK, update the optimum solution
                const auto milis = timer.milliseconds();
                const auto config = nano::append(sconfig, "lambda", lacc.lambda());
                const auto ret = result.update(state, {milis, ++epoch, train, valid, test}, config);

                log_info()
                        << "[" << epoch << "/" << epochs
                        << ":train=" << train
                        << ",valid=" << valid << "|" << nano::to_string(ret)
                        << ",test=" << test
                        << "," << config << ",batch=" << (it.end() - it.begin())
                        << ",g=" << state.g.lpNorm<Eigen::Infinity>()
                        << "] " << timer.elapsed() << ".";

                return !nano::is_done(ret, policy);
        }

        ///
        /// \brief generic training loop given a suitable optimization operator.
        ///
        template
        <
                typename toperator      ///< (value_accumulator, gradient_accumulator, starting_point)
        >
        trainer_result_t trainer_loop(
                const model_t& model, const size_t nthreads,
                const loss_t& loss, const criterion_t& criterion,
                const toperator& trainer)
        {
                vector_t x0;
                model.save_params(x0);

                // setup accumulators
                accumulator_t lacc(model, loss, criterion, criterion_t::type::value);
                accumulator_t gacc(model, loss, criterion, criterion_t::type::vgrad);

                lacc.set_threads(nthreads);
                gacc.set_threads(nthreads);

                // tune the regularization factor (if needed)
                const auto op = [&] (const scalar_t lambda)
                {
                        lacc.set_lambda(lambda);
                        gacc.set_lambda(lambda);
                        return trainer(lacc, gacc, x0);
                };

                if (lacc.can_regularize())
                {
                        const auto space = nano::make_log10_space(scalar_t(-5.0), scalar_t(-1.0), scalar_t(0.5), 4);
                        return nano::tune(op, space).optimum();
                }
                else
                {
                        return op(0.0);
                }
        }
}
