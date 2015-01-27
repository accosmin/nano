#define BOOST_TEST_DYN_LINK
#define BOOST_TEST_MODULE "test_sampler"

#include <boost/test/unit_test.hpp>
#include "libnanocv/tasks/task_syn_dots.h"
#include "libnanocv/sampler.h"
#include "libnanocv/util/logger.h"
#include "libnanocv/util/timer.h"

namespace test
{
        using namespace ncv;

        bool check_fold(const samples_t& samples, ncv::fold_t fold)
        {
                return std::find_if(samples.begin(), samples.end(),
                       [&] (const sample_t& sample) { return sample.m_fold != fold; }) == samples.end();
        }
}

BOOST_AUTO_TEST_CASE(test_sampler)
{
        using namespace ncv;

        const size_t n_samples = 1000;
        const size_t n_rand_samples = n_samples / 4;

        syn_dots_task_t task("rows=28,cols=28,color=luma,dims=5,size=" + text::to_string(n_samples));
        BOOST_CHECK_EQUAL(task.load(""), true);

        for (size_t f = 0; f < task.n_folds(); f ++)
        {
                const fold_t train_fold = {f, protocol::train};
                const fold_t test_fold = {f, protocol::test};

                const string_t header = "fold [" + text::to_string(f + 1) + "/" + text::to_string(task.n_folds()) + "]";
                const string_t train_header = header + " protocol [" + text::to_string(protocol::train) + "]";
                const string_t test_header = header + " protocol [" + text::to_string(protocol::test) + "]";

                const ncv::timer_t timer;

                // batch training samples
                sampler_t sampler(task);
                sampler.reset();
                sampler.setup(train_fold);
                sampler.setup(sampler_t::stype::batch);

                const samples_t train_batch_samples = sampler.get();

                // random uniform training samples
                sampler.reset();
                sampler.setup(train_fold);
                sampler.setup(sampler_t::stype::uniform, n_rand_samples);

                const samples_t train_urand_samples = sampler.get();

                // batch testing samples
                sampler.reset();
                sampler.setup(test_fold);
                sampler.setup(sampler_t::stype::batch);

                const samples_t test_batch_samples = sampler.get();

                // random uniform testing samples
                sampler.reset();
                sampler.setup(test_fold);
                sampler.setup(sampler_t::stype::uniform, n_rand_samples);

                const samples_t test_urand_samples = sampler.get();

                log_info() << "fold [" << (f + 1) << "/" << task.n_folds() << "]: sampled in " << timer.elapsed() << ".";

                // check training & testing split
                BOOST_CHECK_EQUAL(train_batch_samples.size() + test_batch_samples.size(), n_samples);

                // check training samples
                BOOST_CHECK(test::check_fold(train_batch_samples, {f, protocol::train}));
                BOOST_CHECK(test::check_fold(train_urand_samples, {f, protocol::train}));

                ncv::print(train_header + " batch", train_batch_samples);
                ncv::print(train_header + " urand", train_urand_samples);

                // check test samples
                BOOST_CHECK(test::check_fold(test_batch_samples, {f, protocol::test}));
                BOOST_CHECK(test::check_fold(test_urand_samples, {f, protocol::test}));

                ncv::print(test_header + " batch", test_batch_samples);
                ncv::print(test_header + " urand", test_urand_samples);
        }
}
