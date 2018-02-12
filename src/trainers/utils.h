#pragma once

#include "accumulator.h"
#include "math/numeric.h"
#include "trainer_state.h"

namespace nano
{
        inline size_t epoch_size(const task_t& task, const size_t fold, const size_t batch)
        {
                const auto train_size = task.size({fold, protocol::train});
                return idiv(train_size, batch);
        }

        inline trainer_measurement_t measure(const accumulator_t& acc)
        {
                return {acc.vstats().avg(), acc.estats().avg()};
        }

        inline trainer_measurement_t measure(const vector_t& params, const task_t& task, const fold_t& fold, accumulator_t& acc)
        {
                acc.params(params);
                acc.mode(accumulator_t::type::value);
                acc.update(task, fold);
                return measure(acc);
        }
}
