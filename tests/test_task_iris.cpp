#include <set>
#include "task.h"
#include "utest.h"
#include "math/epsilon.h"

using namespace nano;

NANO_BEGIN_MODULE(test_iris)

NANO_CASE(failed)
{
        const auto task = get_tasks().get("iris");
        NANO_REQUIRE(task);

        task->from_json(to_json("path", "/dev/null?!"));
        NANO_CHECK(!task->load());
}

NANO_CASE(loading)
{
        const auto idims = tensor3d_dim_t{4, 1, 1};
        const auto odims = tensor3d_dim_t{3, 1, 1};
        const auto target_sum = scalar_t(2) - static_cast<scalar_t>(nano::size(odims));

        const auto folds = size_t(10);
        const auto samples = size_t(150);

        const auto task = nano::get_tasks().get("iris");
        NANO_REQUIRE(task);
        task->from_json(to_json("folds", folds));
        NANO_REQUIRE(task->load());

        // check dimensions
        NANO_CHECK_EQUAL(task->idims(), idims);
        NANO_CHECK_EQUAL(task->odims(), odims);

        // check folds
        NANO_CHECK_EQUAL(task->fsize(), folds);
        NANO_CHECK_EQUAL(task->size(), folds * samples);

        for (size_t f = 0; f < folds; ++ f)
        {
                const auto train_fold = fold_t{f, protocol::train};
                const auto valid_fold = fold_t{f, protocol::valid};
                const auto test_fold = fold_t{f, protocol::test};

                NANO_CHECK_GREATER(task->size(train_fold), 1 * samples / 10);
                NANO_CHECK_GREATER(task->size(valid_fold), 1 * samples / 10);
                NANO_CHECK_GREATER(task->size(test_fold),  1 * samples / 10);

                NANO_CHECK_EQUAL(task->size(train_fold) + task->size(valid_fold) + task->size(test_fold), samples);
        }

        // check samples
        for (size_t f = 0; f < folds; ++ f)
        {
                const auto train_fold = fold_t{f, protocol::train};
                const auto valid_fold = fold_t{f, protocol::valid};
                const auto test_fold = fold_t{f, protocol::test};

                for (const auto fold : {train_fold, valid_fold, test_fold})
                {
                        std::set<string_t> labels;

                        const auto size = task->size(fold);
                        for (size_t i = 0; i < size; ++ i)
                        {
                                const auto sample = task->get(fold, i, i + 1);
                                const auto& input = sample.idata(0);
                                const auto& label = sample.label(0);
                                const auto& target = sample.odata(0);

                                NANO_CHECK_EQUAL(input.dims(), idims);
                                NANO_CHECK_EQUAL(target.dims(), odims);
                                NANO_CHECK_CLOSE(target.vector().sum(), target_sum, epsilon0<scalar_t>());

                                labels.insert(label);
                        }

                        NANO_CHECK_EQUAL(labels.size(), static_cast<size_t>(nano::size(odims)));
                }
        }

        const size_t max_duplicates = 0;
        NANO_CHECK_LESS_EQUAL(nano::check_duplicates(*task), max_duplicates);
        NANO_CHECK_LESS_EQUAL(nano::check_intersection(*task), max_duplicates);
}

NANO_END_MODULE()
