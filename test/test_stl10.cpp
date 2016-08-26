#include <set>
#include "nano.h"
#include "utest.hpp"
#include "task_util.h"
#include "math/epsilon.hpp"

NANO_BEGIN_MODULE(test_stl10)

NANO_CASE(construction)
{
        using namespace nano;

        const auto path = string_t(std::getenv("HOME")) + "/experiments/databases/stl10";

        const auto idims = 3;
        const auto irows = 96;
        const auto icols = 96;
        const auto osize = 10;
        const auto target_sum = scalar_t(2) - static_cast<scalar_t>(osize);

        const auto folds = size_t(10);
        const auto train_samples = size_t(100000 + 1000);
        const auto test_samples = size_t(8000);

        const auto task = nano::get_tasks().get("stl10", "dir=" + path);
        NANO_REQUIRE(task);
        NANO_REQUIRE(task->load());

        // check dimensions
        NANO_CHECK_EQUAL(task->idims(), idims);
        NANO_CHECK_EQUAL(task->irows(), irows);
        NANO_CHECK_EQUAL(task->icols(), icols);
        NANO_CHECK_EQUAL(task->osize(), osize);

        // check folds
        NANO_CHECK_EQUAL(task->n_folds(), folds);
        NANO_CHECK_EQUAL(task->n_samples(), (train_samples + test_samples) * folds);

        for (size_t f = 0; f < folds; ++ f)
        {
                const auto train_fold = fold_t{f, protocol::train};
                const auto valid_fold = fold_t{f, protocol::valid};
                const auto test_fold = fold_t{f, protocol::test};

                NANO_CHECK_GREATER(task->n_samples(train_fold), train_samples / 10);
                NANO_CHECK_GREATER(task->n_samples(valid_fold), train_samples / 10);
                NANO_CHECK_EQUAL(task->n_samples(train_fold) + task->n_samples(valid_fold), train_samples);

                NANO_CHECK_EQUAL(task->n_samples(test_fold), test_samples);
        }

        // check samples
        std::set<string_t> labels;
        for (size_t f = 0; f < folds; ++ f)
        {
                const auto train_fold = fold_t{f, protocol::train};
                const auto valid_fold = fold_t{f, protocol::valid};
                const auto test_fold = fold_t{f, protocol::test};

                for (const auto fold : {train_fold, valid_fold, test_fold})
                {
                        const auto size = task->n_samples(fold);
                        for (size_t i = 0; i < size; ++ i)
                        {
                                const auto input = task->input(fold, i);
                                const auto label = task->label(fold, i);
                                const auto target = task->target(fold, i);

                                NANO_CHECK_EQUAL(input.size<0>(), idims);
                                NANO_CHECK_EQUAL(input.size<1>(), irows);
                                NANO_CHECK_EQUAL(input.size<2>(), icols);

                                if (label.empty())
                                {
                                        NANO_CHECK_EQUAL(target.size(), 0);
                                        continue;
                                }

                                NANO_CHECK_EQUAL(target.size(), osize);
                                NANO_CHECK_CLOSE(target.array().sum(), target_sum, epsilon0<scalar_t>());

                                labels.insert(label);
                        }
                }
        }

        //NANO_CHECK(nano::check_duplicates(*task)); NB: same samples are duplicated!
        //NANO_CHECK(nano::check_intersection(*task)); NB: same samples are duplicated!
        NANO_CHECK_EQUAL(labels.size(), osize);
}

NANO_END_MODULE()
