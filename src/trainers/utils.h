#pragma once

#include "accumulator.h"

namespace nano
{
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
