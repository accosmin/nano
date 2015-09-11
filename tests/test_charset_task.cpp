#define BOOST_TEST_DYN_LINK
#define BOOST_TEST_MODULE "test_charset_task"

#include <boost/test/unit_test.hpp>
#include "libnanocv/tasks/task_charset.h"

BOOST_AUTO_TEST_CASE(test_charset_task)
{
        using namespace ncv;

        // <charset, color mode, number of outputs/classes/characters>
        std::vector<std::tuple<charset, color_mode, size_t>> configs;
        configs.emplace_back(charset::numeric,             color_mode::luma,       size_t(10));
        configs.emplace_back(charset::lalphabet,           color_mode::rgba,       size_t(26));
        configs.emplace_back(charset::ualphabet,           color_mode::luma,       size_t(26));
        configs.emplace_back(charset::alphabet,            color_mode::luma,       size_t(52));
        configs.emplace_back(charset::alphanumeric,        color_mode::rgba,       size_t(62));

        for (const auto& config : configs)
        {
                const auto type = std::get<0>(config);
                const auto mode = std::get<1>(config);
                const auto irows = size_t(27);
                const auto icols = size_t(23);
                const auto osize = std::get<2>(config);
                const auto count = size_t(10 * osize);
                const auto fsize = size_t(1);   // folds

                charset_task_t task(type, irows, icols, mode, count);

                BOOST_CHECK_EQUAL(task.load(""), true);
                BOOST_CHECK_EQUAL(task.irows(), irows);
                BOOST_CHECK_EQUAL(task.icols(), icols);
                BOOST_CHECK_EQUAL(task.osize(), osize);
                BOOST_CHECK_EQUAL(task.fsize(), fsize);
                BOOST_CHECK_EQUAL(task.color(), mode);
                BOOST_CHECK_EQUAL(task.n_images(), count);
                BOOST_CHECK_EQUAL(task.samples().size(), count);
                BOOST_CHECK_EQUAL(task.sample_size(), rect_t(0, 0, icols, irows));
                BOOST_CHECK_EQUAL(task.labels().size(), osize);

                for (size_t i = 0; i < task.n_images(); i ++)
                {
                        BOOST_CHECK_EQUAL(task.image(i).mode(), mode);
                        BOOST_CHECK_EQUAL(task.image(i).rows(), irows);
                        BOOST_CHECK_EQUAL(task.image(i).cols(), icols);
                }
        }
}
