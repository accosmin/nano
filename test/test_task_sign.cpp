#include "utest.h"
#include "math/epsilon.h"
#include "text/to_params.h"
#include "tasks/task_sign.h"

using namespace nano;

NANO_BEGIN_MODULE(test_sign)

NANO_CASE(construction)
{
        const auto isize = tensor_size_t(13);
        const auto osize = tensor_size_t(14);
        const auto count = tensor_size_t(1001);
        const auto noise = 0;

        const auto idims = dim3d_t{isize, 1, 1};
        const auto odims = dim3d_t{osize, 1, 1};

        sign_task_t task(to_params("isize", isize, "osize", osize, "count", count, "noise", noise));
        NANO_CHECK(task.load());

        NANO_CHECK_EQUAL(task.idims(), idims);
        NANO_CHECK_EQUAL(task.odims(), odims);
        NANO_CHECK_EQUAL(task.n_samples(), count);
        NANO_REQUIRE_EQUAL(task.n_folds(), size_t(1));

        const auto& weights = task.weights();
        const auto& bias = task.bias();

        for (const auto proto : {protocol::train, protocol::valid, protocol::test})
        {
                const auto fold = fold_t{0, proto};
                const auto size = task.n_samples(fold);
                for (size_t i = 0; i < size; ++ i)
                {
                        const auto input = task.input(fold, i);
                        const auto target = task.target(fold, i);

                        NANO_CHECK_EQUAL(input.dims(), idims);
                        NANO_CHECK_EQUAL(target.dims(), odims);
                        NANO_CHECK_GREATER(((weights * input.vector() + bias).array() * target.vector().array()).minCoeff(), 0);
                }
        }
}

NANO_END_MODULE()

