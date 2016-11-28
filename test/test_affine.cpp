#include "utest.h"
#include "math/epsilon.h"
#include "text/to_params.h"
#include "tasks/task_affine.h"

using namespace nano;

const auto idims = tensor_size_t(3);
const auto irows = tensor_size_t(5);
const auto icols = tensor_size_t(7);
const auto osize = tensor_size_t(4);
const auto count = tensor_size_t(1000);
const auto noise = epsilon2<scalar_t>();

NANO_BEGIN_MODULE(test_affine)

NANO_CASE(construction)
{
        affine_task_t task(to_params(
                "idims", idims, "irows", irows, "icols", icols, "osize", osize, "count", count, "noise", noise,
                "mode", affine_mode::regression));

        task.load();

        NANO_CHECK_EQUAL(task.idims(), idims);
        NANO_CHECK_EQUAL(task.irows(), irows);
        NANO_CHECK_EQUAL(task.icols(), icols);
        NANO_CHECK_EQUAL(task.osize(), osize);
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
                        NANO_CHECK_EIGEN_CLOSE(weights * input.vector() + bias, target, 2 * noise);
                }
        }
}

NANO_END_MODULE()
