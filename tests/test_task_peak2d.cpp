#include "task.h"
#include "utest.h"

using namespace nano;

NANO_BEGIN_MODULE(test_task_peak2d)

NANO_CASE(default_config)
{
        const auto task = nano::get_tasks().get("synth-peak2d");
        NANO_REQUIRE(task);

        json_t json;
        task->to_json(json);

        size_t folds = 0;
        from_json(json, "folds", folds);

        NANO_CHECK_EQUAL(folds, 10u);
}

NANO_CASE(construction)
{
        const auto irows = 14;
        const auto icols = 8;
        const auto count = size_t(250);
        const auto folds = size_t(20);

        auto task = get_tasks().get("synth-peak2d");
        NANO_REQUIRE(task);
        task->from_json(to_json("irows", irows, "icols", icols, "noise", 0, "count", count, "folds", folds));
        NANO_CHECK(task->load());
        task->describe("synth-peak2d");

        NANO_CHECK_EQUAL(task->idims(), make_dims(1, irows, icols));
        NANO_CHECK_EQUAL(task->odims(), make_dims(2, 1, 1));
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

                                NANO_CHECK_EQUAL(input.dims(), make_dims(1, irows, icols));
                                NANO_CHECK_EQUAL(target.dims(), make_dims(2, 1, 1));

                                tensor_size_t r = 0, c = 0;
                                const auto min = input.matrix(0).minCoeff(&r, &c);

                                NANO_CHECK_CLOSE(min, 0, epsilon0<scalar_t>());
                                NANO_CHECK_CLOSE(target(0), scalar_t(c) / scalar_t(icols), epsilon0<scalar_t>());
                                NANO_CHECK_CLOSE(target(1), scalar_t(r) / scalar_t(irows), epsilon0<scalar_t>());
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
