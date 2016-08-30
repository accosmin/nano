#pragma once

#include "math/tune.hpp"
#include "accumulator.h"
#include "trainer_result.h"

namespace nano
{
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
                        const auto space = nano::make_log10_space(scalar_t(-5.0), scalar_t(+0.0), scalar_t(0.5), 5);
                        return nano::tune(op, space).optimum();
                }
                else
                {
                        return op(0.0);
                }
        }
}
