#include "utest.h"
#include "math/epsilon.h"
#include "text/to_params.h"
#include "tasks/task_matmul.h"

using namespace nano;

NANO_BEGIN_MODULE(test_matmul)

NANO_CASE(construction)
{
        const auto irows = tensor_size_t(5);
        const auto icols = tensor_size_t(7);
        const auto count = tensor_size_t(1003);
        const auto noise = epsilon2<scalar_t>();

        const auto idims = dim3d_t{2, irows, icols};
        const auto odims = dim3d_t{1, irows, irows};

        matmul_task_t task(to_params("irows", irows, "icols", icols, "count", count, "noise", noise));
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
                        const auto output = weights * (input.matrix(0) * input.matrix(1).transpose()) + bias;
                        NANO_CHECK_EIGEN_CLOSE(output, target.matrix(0), 2 * noise);
                }
        }
}

NANO_END_MODULE()

