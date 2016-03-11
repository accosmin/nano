#define BOOST_TEST_DYN_LINK
#define BOOST_TEST_MODULE "test_charset_task"

#include "unit_test.hpp"
#include "cortex/tasks/task_charset.h"

NANO_BEGIN_MODULE(test_charset_task)

NANO_CASE(evaluate)
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

                charset_task_t task(type, irows, icols, mode, count);

                NANO_CHECK_EQUAL(task.load(""), true);
                NANO_CHECK_EQUAL(task.irows(), irows);
                NANO_CHECK_EQUAL(task.icols(), icols);
                NANO_CHECK_EQUAL(task.osize(), osize);
                NANO_CHECK_EQUAL(task.fsize(), fsize);
                NANO_CHECK_EQUAL(task.color(), mode);
                NANO_CHECK_EQUAL(task.n_images(), count);
                NANO_CHECK_EQUAL(task.samples().size(), count);
                NANO_CHECK_EQUAL(task.sample_size(), rect_t(0, 0, icols, irows));
                NANO_CHECK_EQUAL(task.labels().size(), static_cast<size_t>(osize));

                for (size_t i = 0; i < task.n_images(); ++ i)
                {
                        NANO_CHECK_EQUAL(task.image(i).mode(), mode);
                        NANO_CHECK_EQUAL(task.image(i).rows(), irows);
                        NANO_CHECK_EQUAL(task.image(i).cols(), icols);
                }
        }
}

NANO_END_MODULE()
