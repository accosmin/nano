#include <set>
#include "task.h"
#include "utest.h"
#include "math/epsilon.h"

using namespace nano;

NANO_BEGIN_MODULE(test_svhn)

NANO_CASE(failed)
{
        const auto task = get_tasks().get("svhn");
        NANO_REQUIRE(task);

        task->from_json(to_json("dir", "/dev/null?!"));
        NANO_CHECK(!task->load());
}

NANO_CASE(loading)
{
        const auto idims = tensor3d_dim_t{3, 32, 32};
        const auto odims = tensor3d_dim_t{10, 1, 1};
        const auto target_sum = scalar_t(2) - static_cast<scalar_t>(nano::size(odims));

        const auto folds = size_t(3);
        const auto train_samples = size_t(73257 + 531131);
        const auto test_samples = size_t(26032);

        const auto task = nano::get_tasks().get("svhn");
        NANO_REQUIRE(task);
        task->from_json(to_json("folds", folds));
        NANO_REQUIRE(task->load());
        task->describe("svhn");

        NANO_CHECK_EQUAL(task->idims(), idims);
        NANO_CHECK_EQUAL(task->odims(), odims);
        NANO_CHECK_EQUAL(task->fsize(), folds);
        NANO_CHECK_EQUAL(task->size(), folds * (train_samples + test_samples));

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
                                NANO_CHECK_CLOSE(target.vector().sum(), target_sum, epsilon0<scalar_t>());
                        }

                        NANO_CHECK_EQUAL(task->labels({f, p}).size(), static_cast<size_t>(nano::size(odims)));
                }

                NANO_CHECK_EQUAL(task->size({f, protocol::train}), 483509);
                NANO_CHECK_EQUAL(task->size({f, protocol::valid}), 120879);
                NANO_CHECK_EQUAL(task->size({f, protocol::test}), test_samples);

                NANO_CHECK_EQUAL(
                        task->size({f, protocol::train}) +
                        task->size({f, protocol::valid}) +
                        task->size({f, protocol::test}),
                        task->size() / task->fsize());

                NANO_CHECK_LESS_EQUAL(task->duplicates(f), size_t(2));
                NANO_CHECK_LESS_EQUAL(task->intersections(f), size_t(0));
        }

        NANO_CHECK_LESS_EQUAL(task->duplicates(), size_t(2));
        NANO_CHECK_LESS_EQUAL(task->intersections(), size_t(0));
        NANO_CHECK_EQUAL(task->labels().size(), static_cast<size_t>(nano::size(odims)));
}

NANO_END_MODULE()
