#include "task.h"
#include "utest.h"
#include "math/epsilon.h"

using namespace nano;

static auto get_task(const tensor_size_t isize, const tensor_size_t osize, const scalar_t noise, const size_t count)
{
        auto task = get_tasks().get("synth-affine");
        NANO_REQUIRE(task);
        task->config(json_writer_t().object("isize", isize, "osize", osize, "noise", noise, "count", count).str());
        return task;
}

NANO_BEGIN_MODULE(test_task_affine)

NANO_CASE(construction)
{
        auto task = get_task(11, 13, 0, 132);
        NANO_CHECK(task->load());

        const auto idims = tensor3d_dim_t{11, 1, 1};
        const auto odims = tensor3d_dim_t{13, 1, 1};

        NANO_CHECK_EQUAL(task->idims(), idims);
        NANO_CHECK_EQUAL(task->odims(), odims);
        NANO_CHECK_EQUAL(task->fsize(), size_t(1));
        NANO_CHECK_EQUAL(task->size(), size_t(132));

        NANO_CHECK_EQUAL(
                task->size({0, protocol::train}) +
                task->size({0, protocol::valid}) +
                task->size({0, protocol::test}),
                size_t(132));

        for (const auto p : {protocol::train, protocol::valid, protocol::test})
        {
                const auto size = task->size({0, p});
                for (size_t i = 0; i < size; ++ i)
                {
                        const auto sample = task->get({0, p}, i, i + 1);
                        const auto& input = sample.idata(0);
                        const auto& target = sample.odata(0);

                        NANO_CHECK_EQUAL(input.dims(), idims);
                        NANO_CHECK_EQUAL(target.dims(), odims);
                }
        }

        const size_t max_duplicates = 0;
        NANO_CHECK_LESS_EQUAL(nano::check_duplicates(*task), max_duplicates);
        NANO_CHECK_LESS_EQUAL(nano::check_intersection(*task), max_duplicates);
}

NANO_END_MODULE()
