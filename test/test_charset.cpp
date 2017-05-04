#include "nano.h"
#include "utest.h"
#include "task_util.h"
#include "vision/color.h"
#include "task_iterator.h"
#include "tasks/charset.h"
#include "text/to_params.h"

using namespace nano;

NANO_BEGIN_MODULE(test_charset)

NANO_CASE(construction)
{
        // <charset, color mode, number of outputs/classes/characters>
        std::vector<std::tuple<charset_type, color_mode, tensor_size_t>> configs;
        configs.emplace_back(charset_type::digit,            color_mode::luma,       tensor_size_t(10));
        configs.emplace_back(charset_type::lalpha,           color_mode::rgba,       tensor_size_t(26));
        configs.emplace_back(charset_type::ualpha,           color_mode::luma,       tensor_size_t(26));
        configs.emplace_back(charset_type::alpha,            color_mode::luma,       tensor_size_t(52));
        configs.emplace_back(charset_type::alphanum,         color_mode::rgba,       tensor_size_t(62));

        for (const auto& config : configs)
        {
                const auto type = std::get<0>(config);
                const auto mode = std::get<1>(config);
                const auto irows = tensor_size_t(17);
                const auto icols = tensor_size_t(16);
                const auto osize = std::get<2>(config);
                const auto count = size_t(10 * osize);
                const auto fsize = size_t(1);   // folds

                const auto idims = tensor3d_dims_t{(mode == color_mode::rgba) ? 4 : 1, irows, icols};
                const auto odims = tensor3d_dims_t{osize, 1, 1};

                auto task = get_tasks().get("synth-charset", to_params(
                        "type", type, "color", mode, "irows", irows, "icols", icols, "count", count));

                NANO_CHECK_EQUAL(task->load(), true);
                NANO_CHECK_EQUAL(task->idims(), idims);
                NANO_CHECK_EQUAL(task->odims(), odims);
                NANO_CHECK_EQUAL(task->fsize(), fsize);
                NANO_CHECK_EQUAL(task->size(), count);
        }
}

NANO_CASE(fixed_batch_iterator)
{
        auto task = get_tasks().get("synth-charset", to_params(
                "type", charset_type::digit, "color", color_mode::rgba, "irows", 16, "icols", 16, "count", 10000));

        NANO_CHECK_EQUAL(task->load(), true);

        const auto batch = size_t(123);
        const auto fold = fold_t{0, protocol::train};
        const auto fold_size = task->size(fold);

        task_iterator_t it(*task, fold, batch);
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

        task_iterator_t it(*task, fold, batch0, factor);
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

NANO_CASE(from_params)
{
        auto task = get_tasks().get("synth-charset", "type=alpha,color=rgb,irows=23,icols=29,count=102");
        NANO_CHECK(task->load());

        const auto idims = tensor3d_dims_t{3, 23, 29};
        const auto odims = tensor3d_dims_t{52, 1, 1};

        NANO_CHECK_EQUAL(task->idims(), idims);
        NANO_CHECK_EQUAL(task->odims(), odims);
        NANO_CHECK_EQUAL(task->fsize(), size_t(1));
        NANO_CHECK_EQUAL(task->size(), size_t(102));

        NANO_CHECK_EQUAL(
                task->size({0, protocol::train}) +
                task->size({0, protocol::valid}) +
                task->size({0, protocol::test}),
                size_t(102));

        for (const auto p : {protocol::train, protocol::valid, protocol::test})
        {
                const auto size = task->size({0, p});
                for (size_t i = 0; i < size; ++ i)
                {
                        const auto input = task->input({0, p}, i);
                        const auto target = task->target({0, p}, i);

                        NANO_CHECK_EQUAL(input.dims(), idims);
                        NANO_CHECK_EQUAL(target.dims(), odims);
                }
        }

        const size_t max_duplicates = 0;
        NANO_CHECK_LESS_EQUAL(nano::check_duplicates(*task), max_duplicates);
        NANO_CHECK_LESS_EQUAL(nano::check_intersection(*task), max_duplicates);
}

NANO_END_MODULE()
