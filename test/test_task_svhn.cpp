#include <set>
#include "utest.h"
#include "task_util.h"
#include "math/epsilon.h"

NANO_BEGIN_MODULE(test_svhn)

NANO_CASE(construction)
{
        using namespace nano;

        const auto idims = tensor3d_dims_t{3, 32, 32};
        const auto odims = tensor3d_dims_t{10, 1, 1};
        const auto target_sum = scalar_t(2) - static_cast<scalar_t>(nano::size(odims));

        const auto folds = size_t(1);
        const auto train_samples = size_t(73257 + 531131);
        const auto test_samples = size_t(26032);
        const auto train_fold = fold_t{0, protocol::train};
        const auto valid_fold = fold_t{0, protocol::valid};
        const auto test_fold = fold_t{0, protocol::test};

        const auto task = nano::get_tasks().get("svhn");
        NANO_REQUIRE(task);
        NANO_REQUIRE(task->load());

        // check dimensions
        NANO_CHECK_EQUAL(task->idims(), idims);
        NANO_CHECK_EQUAL(task->odims(), odims);

        // check folds
        NANO_CHECK_EQUAL(task->fsize(), folds);
        NANO_CHECK_EQUAL(task->size(), train_samples + test_samples);

        NANO_CHECK_GREATER(task->size(train_fold), train_samples / 10);
        NANO_CHECK_GREATER(task->size(valid_fold), train_samples / 10);
        NANO_CHECK_EQUAL(task->size(train_fold) + task->size(valid_fold), train_samples);

        NANO_CHECK_EQUAL(task->size(test_fold), test_samples);

        // check samples
        std::set<string_t> labels;
        for (const auto fold : {train_fold, valid_fold, test_fold})
        {
                const auto size = task->size(fold);
                for (size_t i = 0; i < size; ++ i)
                {
                        const auto input = task->input(fold, i);
                        const auto label = task->label(fold, i);
                        const auto target = task->target(fold, i);

                        NANO_CHECK_EQUAL(input.dims(), idims);
                        NANO_CHECK_EQUAL(target.dims(), odims);
                        NANO_CHECK_CLOSE(target.vector().sum(), target_sum, epsilon0<scalar_t>());

                        labels.insert(label);
                }
        }

        const size_t max_duplicates = 2;
        NANO_CHECK_LESS_EQUAL(nano::check_duplicates(*task), max_duplicates);
        NANO_CHECK_LESS_EQUAL(nano::check_intersection(*task), max_duplicates);
        NANO_CHECK_EQUAL(labels.size(), static_cast<size_t>(nano::size(odims)));
}

NANO_END_MODULE()
