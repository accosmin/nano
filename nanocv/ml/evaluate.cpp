#include "task.h"
#include "sampler.h"
#include "evaluate.h"
#include "accumulator.h"

namespace ncv
{
        size_t evaluate(const task_t& task, const fold_t& fold, const loss_t& loss, const model_t& model,
                scalar_t& lvalue, scalar_t& lerror)
        {
                sampler_t sampler(task.samples());
                sampler.setup(fold).setup(sampler_t::atype::annotated);

                accumulator_t accumulator(model, 0, "avg", criterion_t::type::value, 0.0);
                accumulator.update(task, sampler.get(), loss);

                lvalue = accumulator.value();
                lerror = accumulator.avg_error();

                return accumulator.count();
        }
}

