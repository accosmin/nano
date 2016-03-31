#pragma once

#include "protocol.h"
#include "math/tune.hpp"
#include "accumulator.h"
#include "trainer_result.h"

namespace nano
{
        class task_t;

        ///
        /// \brief generic training loop given a suitable optimization operator.
        ///
        template
        <
                typename toperator      ///< (task, tfold, vfold, value_accumulator, gradient_accumulator, start, verbose)
        >
        trainer_result_t trainer_loop(
                const model_t& model, const task_t& task, const size_t fold, const size_t nthreads,
                const loss_t& loss, const criterion_t& criterion,
                const toperator& trainer)
        {
                vector_t x0;
                model.save_params(x0);

                const auto train_fold = fold_t{fold, protocol::train};
                const auto valid_fold = fold_t{fold, protocol::valid};
                const auto test_fold = fold_t{fold, protocol::test};

                // setup accumulators
                accumulator_t lacc(model, loss, criterion, criterion_t::type::value); lacc.set_threads(nthreads);
                accumulator_t gacc(model, loss, criterion, criterion_t::type::vgrad); gacc.set_threads(nthreads);

                // tune the regularization factor (if needed)
                const auto op = [&] (const scalar_t lambda)
                {
                        lacc.set_lambda(lambda);
                        gacc.set_lambda(lambda);
                        return trainer(train_fold, valid_fold, lacc, gacc, x0);
                };

                trainer_result_t result;
                if (lacc.can_regularize())
                {
                        const auto space = nano::make_log10_space(-6.0, +6.0, 0.5);
                        result = nano::tune(op, space).optimum();
                }
                else
                {
                        result = op(0.0);
                }

                // compute the test error

                // \todo: store the test error in result
                // \todo: update apps/trainer.cpp to read directly the test error from trainer_result_t
                // \todo: update apps/benchmark_trainers.cpp to print the test error
                // \todo: update apps/benchmark_trainers.cpp to vary the criterion for the SAME starting parameters for a fair comparison

                //lacc.set_lambda(0.0);
                //lacc.set_params(result.optimum_params);
                //lacc.update(task, test_fold);


                // OK
                return result;
        }
}
