#include "utest.h"
#include "vision/color.h"
#include "tasks/charset.h"
#include "trainers/minibatch.h"

using namespace nano;

NANO_BEGIN_MODULE(test_minibatch)

NANO_CASE(fixed_batch_iterator)
{
        auto task = get_tasks().get("synth-charset", to_params(
                "type", charset_type::digit, "color", color_mode::rgba, "irows", 16, "icols", 16, "count", 10000));

        NANO_CHECK_EQUAL(task->load(), true);

        const auto batch = size_t(123);
        const auto fold = fold_t{0, protocol::train};
        const auto fold_size = task->size(fold);

        minibatch_t it(*task, fold, batch);
        for (size_t i = 0; i < 1000; ++ i)
        {
                NANO_CHECK_LESS(it.begin(), it.end());
                NANO_CHECK_LESS_EQUAL(it.end(), fold_size);
                NANO_CHECK_LESS_EQUAL(it.end(), it.begin() + batch);

                const auto end = it.end();
                it.next();

                if (end + batch >= fold_size)
                {
                        NANO_CHECK_EQUAL(it.begin(), size_t(0));
                        NANO_CHECK_EQUAL(it.end(), batch);
                }
                else
                {
                        NANO_CHECK_EQUAL(end, it.begin());
                }
        }
}

NANO_CASE(increasing_batch_iterator)
{
        auto task = get_tasks().get("synth-charset", to_params(
                "type", charset_type::digit, "color", color_mode::rgba, "irows", 16, "icols", 16, "count", 10000));

        NANO_CHECK_EQUAL(task->load(), true);

        const auto batch0 = size_t(3);
        const auto factor = scalar_t(1.05);
        const auto fold = fold_t{0, protocol::train};
        const auto fold_size = task->size(fold);

        minibatch_t it(*task, fold, batch0, factor);
        for (size_t i = 0; i < 1000; ++ i)
        {
                NANO_CHECK_LESS(it.begin(), it.end());
                NANO_CHECK_LESS_EQUAL(it.end(), fold_size);
                NANO_CHECK_GREATER_EQUAL(it.end(), it.begin() + batch0);

                const auto begin = it.begin();
                const auto end = it.end();
                it.next();

                if (end + (end - begin) >= fold_size)
                {
                        NANO_CHECK_EQUAL(it.begin(), size_t(0));
                }
        }
}

NANO_END_MODULE()
