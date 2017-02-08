#include <set>
#include "nano.h"
#include "utest.h"
#include "task_util.h"
#include "math/epsilon.h"

NANO_BEGIN_MODULE(test_cifar10)

NANO_CASE(construction)
{
        using namespace nano;

        const auto path = string_t(std::getenv("HOME")) + "/experiments/databases/cifar10";

        const auto idims = dim3d_t{3, 32, 32};
        const auto odims = dim3d_t{10, 1, 1};
        const auto target_sum = scalar_t(2) - static_cast<scalar_t>(odims.size());

        const auto folds = size_t(1);
        const auto train_samples = 50000;
        const auto test_samples = 10000;
        const auto train_fold = fold_t{0, protocol::train};
        const auto valid_fold = fold_t{0, protocol::valid};
        const auto test_fold = fold_t{0, protocol::test};

        const auto task = nano::get_tasks().get("cifar10", "dir=" + path);
        NANO_REQUIRE(task);
        NANO_REQUIRE(task->load());

        // check dimensions
        NANO_CHECK_EQUAL(task->idims(), idims);
        NANO_CHECK_EQUAL(task->odims(), odims);

        // check folds
        NANO_CHECK_EQUAL(task->n_folds(), folds);
        NANO_CHECK_EQUAL(task->n_samples(), train_samples + test_samples);

        NANO_CHECK_GREATER(task->n_samples(train_fold), train_samples / 10);
        NANO_CHECK_GREATER(task->n_samples(valid_fold), train_samples / 10);
        NANO_CHECK_EQUAL(task->n_samples(train_fold) + task->n_samples(valid_fold), train_samples);

        NANO_CHECK_EQUAL(task->n_samples(test_fold), test_samples);

        // check samples
        std::set<string_t> labels;
        for (const auto fold : {train_fold, valid_fold, test_fold})
        {
                const auto size = task->n_samples(fold);
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

        const size_t max_duplicates = 0;
        NANO_CHECK_LESS_EQUAL(nano::check_duplicates(*task), max_duplicates);
        NANO_CHECK_LESS_EQUAL(nano::check_intersection(*task), max_duplicates);
        NANO_CHECK_EQUAL(labels.size(), static_cast<size_t>(odims.size()));
}

NANO_END_MODULE()
