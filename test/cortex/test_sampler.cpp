#define BOOST_TEST_DYN_LINK
#define BOOST_TEST_MODULE "test_sampler"

#include <boost/test/unit_test.hpp>
#include "cortex/timer.h"
#include "cortex/logger.h"
#include "cortex/sampler.h"
#include "text/to_string.hpp"
#include "cortex/tasks/task_charset.h"

namespace test
{
        using namespace cortex;

        bool check_fold(const samples_t& samples, fold_t fold)
        {
                const auto op = [=] (const sample_t& sample) { return sample.m_fold != fold; };
                return std::find_if(samples.begin(), samples.end(), op) == samples.end();
        }
}

BOOST_AUTO_TEST_CASE(test_sampler)
{
        using namespace cortex;

        const size_t n_samples = 1000;
        const size_t n_rand_samples = n_samples / 4;

        charset_task_t task(charset::numeric, 28, 28, color_mode::luma, n_samples);
        BOOST_CHECK_EQUAL(task.load(""), true);

        for (size_t f = 0; f < task.fsize(); f ++)
        {
                const fold_t train_fold = {f, protocol::train};
                const fold_t test_fold = {f, protocol::test};

                const string_t header = "fold [" + text::to_string(f + 1) + "/" + text::to_string(task.fsize()) + "]";
                const string_t train_header = header + " protocol [" + text::to_string(protocol::train) + "]";
                const string_t test_header = header + " protocol [" + text::to_string(protocol::test) + "]";

                const cortex::timer_t timer;

                // batch training samples
                const auto train_batch_samples =
                        sampler_t(task.samples()).push(train_fold).get();

                // random uniform training samples
                const auto train_urand_samples =
                        sampler_t(task.samples()).push(train_fold).push(n_rand_samples).get();

                // batch testing samples
                const auto test_batch_samples =
                        sampler_t(task.samples()).push(test_fold).get();

                // random uniform testing samples
                const auto test_urand_samples =
                        sampler_t(task.samples()).push(test_fold).push(n_rand_samples).get();

                log_info() << "fold [" << (f + 1) << "/" << task.fsize() << "]: sampled in " << timer.elapsed() << ".";

                // check training & testing split
                BOOST_CHECK_EQUAL(train_batch_samples.size() + test_batch_samples.size(), n_samples);

                // check training samples
                BOOST_CHECK(test::check_fold(train_batch_samples, train_fold));
                BOOST_CHECK(test::check_fold(train_urand_samples, train_fold));

                cortex::print(train_header + " batch", train_batch_samples);
                cortex::print(train_header + " urand", train_urand_samples);

                // check test samples
                BOOST_CHECK(test::check_fold(test_batch_samples, test_fold));
                BOOST_CHECK(test::check_fold(test_urand_samples, test_fold));

                cortex::print(test_header + " batch", test_batch_samples);
                cortex::print(test_header + " urand", test_urand_samples);
        }
}
