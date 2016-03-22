#include "unit_test.hpp"
#include "cortex/tasks/task_charset.h"

NANO_BEGIN_MODULE(test_charset)

NANO_CASE(construction)
{
        using namespace nano;

        // <charset, color mode, number of outputs/classes/characters>
        std::vector<std::tuple<charset, color_mode, tensor_size_t>> configs;
        configs.emplace_back(charset::numeric,             color_mode::luma,       tensor_size_t(10));
        configs.emplace_back(charset::lalphabet,           color_mode::rgba,       tensor_size_t(26));
        configs.emplace_back(charset::ualphabet,           color_mode::luma,       tensor_size_t(26));
        configs.emplace_back(charset::alphabet,            color_mode::luma,       tensor_size_t(52));
        configs.emplace_back(charset::alphanumeric,        color_mode::rgba,       tensor_size_t(62));

        for (const auto& config : configs)
        {
                const auto type = std::get<0>(config);
                const auto mode = std::get<1>(config);
                const auto irows = size_t(17);
                const auto icols = size_t(16);
                const auto osize = std::get<2>(config);
                const auto count = size_t(10 * osize);
                const auto fsize = size_t(1);   // folds

                charset_task_t task(type, mode, irows, icols, count);

                NANO_CHECK_EQUAL(task.load(""), true);
                NANO_CHECK_EQUAL(task.irows(), irows);
                NANO_CHECK_EQUAL(task.icols(), icols);
                NANO_CHECK_EQUAL(task.osize(), osize);
                NANO_CHECK_EQUAL(task.n_folds(), fsize);
                NANO_CHECK_EQUAL(task.color(), mode);
                NANO_CHECK_EQUAL(task.n_images(), count);
                NANO_CHECK_EQUAL(task.n_samples(), count);

                for (size_t i = 0; i < task.n_images(); ++ i)
                {
                        NANO_CHECK_EQUAL(task.image(i).mode(), mode);
                        NANO_CHECK_EQUAL(task.image(i).rows(), irows);
                        NANO_CHECK_EQUAL(task.image(i).cols(), icols);
                }
        }
}

NANO_CASE(from_params)
{
        using namespace nano;

        charset_task_t task("type=alpha,color=luma,irows=23,icols=29,count=102");
        NANO_CHECK(task.load());

        NANO_CHECK_EQUAL(task.irows(), 23);
        NANO_CHECK_EQUAL(task.icols(), 29);
        NANO_CHECK_EQUAL(task.idims(), 1);
        NANO_CHECK_EQUAL(task.osize(), 52);
        NANO_CHECK_EQUAL(task.n_folds(), size_t(1));
        NANO_CHECK_EQUAL(task.n_samples(), size_t(102));

        NANO_CHECK_EQUAL(
                task.n_samples({0, protocol::train}) +
                task.n_samples({0, protocol::valid}) +
                task.n_samples({0, protocol::test}),
                size_t(102));

        for (const auto p : {protocol::train, protocol::valid, protocol::test})
        {
                const auto size = task.n_samples({0, p});
                for (size_t i = 0; i < size; ++ i)
                {
                        const auto input = task.input({0, p}, i);
                        const auto target = task.target({0, p}, i);

                        NANO_CHECK_EQUAL(input.size<0>(), 1);
                        NANO_CHECK_EQUAL(input.size<1>(), 23);
                        NANO_CHECK_EQUAL(input.size<2>(), 29);
                        NANO_CHECK_EQUAL(target.size(), 52);
                }
        }
}

NANO_END_MODULE()
