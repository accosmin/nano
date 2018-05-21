#include "task.h"
#include "utest.h"
#include "math/epsilon.h"

using namespace nano;

NANO_BEGIN_MODULE(test_task_affine)

NANO_CASE(construction)
{
        const auto isize = 11;
        const auto osize = 13;
        const auto count = size_t(350);
        const auto folds = size_t(7);

        auto task = get_tasks().get("synth-affine");
        NANO_REQUIRE(task);
        task->from_json(to_json("isize", isize, "osize", osize, "noise", 0, "count", count, "folds", folds));
        NANO_CHECK(task->load());
        task->describe("synth-affine");

        NANO_CHECK_EQUAL(task->idims(), make_dims(isize, 1, 1));
        NANO_CHECK_EQUAL(task->odims(), make_dims(osize, 1, 1));
        NANO_CHECK_EQUAL(task->fsize(), folds);
        NANO_CHECK_EQUAL(task->size(), folds * count);

        for (size_t f = 0; f < task->fsize(); ++ f)
        {
                for (const auto p : {protocol::train, protocol::valid, protocol::test})
                {
                        for (size_t i = 0, size = task->size({f, p}); i < size; ++ i)
                        {
                                const auto sample = task->get({f, p}, i, i + 1);
                                const auto& input = sample.idata(0);
                                const auto& target = sample.odata(0);

                                NANO_CHECK_EQUAL(input.dims(), make_dims(isize, 1, 1));
                                NANO_CHECK_EQUAL(target.dims(), make_dims(osize, 1, 1));
                        }
                }

                NANO_CHECK_EQUAL(task->size({f, protocol::train}), 40 * count / 100);
                NANO_CHECK_EQUAL(task->size({f, protocol::valid}), 30 * count / 100);
                NANO_CHECK_EQUAL(task->size({f, protocol::test}), 30 * count / 100);

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
