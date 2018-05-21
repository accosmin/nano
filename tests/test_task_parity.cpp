#include "task.h"
#include "utest.h"
#include "math/epsilon.h"

using namespace nano;

NANO_BEGIN_MODULE(test_task_parity)

NANO_CASE(construction)
{
        auto task = get_tasks().get("synth-parity");
        NANO_REQUIRE(task);
        task->from_json(to_json("n", 11, "count", 132));
        NANO_CHECK(task->load());

        const auto idims = tensor3d_dim_t{11, 1, 1};
        const auto odims = tensor3d_dim_t{1, 1, 1};

        NANO_CHECK_EQUAL(task->idims(), idims);
        NANO_CHECK_EQUAL(task->odims(), odims);
        NANO_CHECK_EQUAL(task->fsize(), size_t(1));
        NANO_CHECK_EQUAL(task->size(), size_t(132));

        for (size_t f = 0; f < task->fsize(); ++ f)
        {
                for (const auto p : {protocol::train, protocol::valid, protocol::test})
                {
                        for (size_t i = 0, size = task->size({f, p}); i < size; ++ i)
                        {
                                const auto sample = task->get({f, p}, i, i + 1);
                                const auto& input = sample.idata(0);
                                const auto& target = sample.odata(0);

                                NANO_CHECK_EQUAL(input.dims(), idims);
                                NANO_CHECK_EQUAL(target.dims(), odims);

                                size_t ones = 0;
                                for (auto x = 0; x < input.size(); ++ x)
                                {
                                        if (input(x) > scalar_t(0.5))
                                        {
                                                NANO_CHECK_CLOSE(input(x), scalar_t(1), epsilon0<scalar_t>());
                                                ++ ones;
                                        }
                                        else
                                        {
                                                NANO_CHECK_CLOSE(input(x), scalar_t(0), epsilon0<scalar_t>());
                                        }
                                }

                                const auto expected_target = (ones % 2) ? pos_target() : neg_target();
                                NANO_CHECK_CLOSE(target(0), expected_target, epsilon0<scalar_t>());
                        }
                }

                NANO_CHECK_EQUAL(
                        task->size({f, protocol::train}) +
                        task->size({f, protocol::valid}) +
                        task->size({f, protocol::test}),
                        task->size() / task->fsize());

                NANO_CHECK_LESS_EQUAL(task->duplicates(f), size_t(0));
                NANO_CHECK_LESS_EQUAL(task->intersections(f), size_t(0));
        }

        NANO_CHECK_LESS_EQUAL(task->duplicates(), size_t(0));
        NANO_CHECK_LESS_EQUAL(task->intersections(), size_t(0));
}

NANO_END_MODULE()
