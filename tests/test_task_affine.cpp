#include "task.h"
#include "utest.h"
#include "math/epsilon.h"

using namespace nano;

NANO_BEGIN_MODULE(test_task_affine)

NANO_CASE(regression)
{
        const auto isize = 11;
        const auto osize = 13;
        const auto count = 132;

        const auto config = json_writer_t().object(
                "type", "regression", "isize", isize, "osize", osize, "noise", 0, "count", count).str();

        auto task = get_tasks().get("synth-affine");
        NANO_REQUIRE(task);
        task->config(config);
        NANO_CHECK(task->load());

        NANO_CHECK_EQUAL(task->idims(), make_dims(isize, 1, 1));
        NANO_CHECK_EQUAL(task->odims(), make_dims(osize, 1, 1));
        NANO_CHECK_EQUAL(task->fsize(), size_t(1));
        NANO_CHECK_EQUAL(task->size(), size_t(count));

        NANO_CHECK_EQUAL(
                task->size({0, protocol::train}) +
                task->size({0, protocol::valid}) +
                task->size({0, protocol::test}),
                size_t(count));

        for (const auto p : {protocol::train, protocol::valid, protocol::test})
        {
                const auto size = task->size({0, p});
                for (size_t i = 0; i < size; ++ i)
                {
                        const auto sample = task->get({0, p}, i, i + 1);
                        const auto& input = sample.idata(0);
                        const auto& target = sample.odata(0);

                        NANO_CHECK_EQUAL(input.dims(), make_dims(isize, 1, 1));
                        NANO_CHECK_EQUAL(target.dims(), make_dims(osize, 1, 1));
                }
        }

        const size_t max_duplicates = 0;
        NANO_CHECK_LESS_EQUAL(nano::check_duplicates(*task), max_duplicates);
        NANO_CHECK_LESS_EQUAL(nano::check_intersection(*task), max_duplicates);
}

NANO_END_MODULE()
