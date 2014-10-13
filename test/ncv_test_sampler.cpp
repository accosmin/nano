#include "nanocv.h"
#include "tasks/task_dummy.h"
#include <algorithm>

using namespace ncv;

bool check_samples(const samples_t& samples, ncv::fold_t fold)
{
        return std::find_if(samples.begin(), samples.end(),
               [&] (const sample_t& sample) { return sample.m_fold != fold; }) == samples.end();
}

int main(int argc, char *argv[])
{
        ncv::init();

        const size_t n_samples = 1000;
        const size_t n_rand_samples = n_samples / 4;

        dummy_task_t task;
        task.set_rows(28);
        task.set_cols(28);
        task.set_color(color_mode::luma);
        task.set_outputs(5);
        task.set_folds(3);
        task.set_size(n_samples);
        task.setup();

        for (size_t f = 0; f < task.n_folds(); f ++)
        {
                const ncv::timer_t timer;

                const fold_t train_fold = {f, protocol::train};
                const fold_t test_fold = {f, protocol::test};

                const string_t header = "fold [" + text::to_string(f + 1) + "/" + text::to_string(task.n_folds()) + "]";
                const string_t train_header = header + " protocol [" + text::to_string(protocol::train) + "]";
                const string_t test_header = header + " protocol [" + text::to_string(protocol::test) + "]";

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

                // check results
                if (train_batch_samples.size() + test_batch_samples.size() != n_samples)
                {
                        log_error() << "invalid number of batch samples "
                                    << (train_batch_samples.size() + test_batch_samples.size())
                                    << "/" << n_samples << "!";
                }

                if (!check_samples(train_batch_samples, {f, protocol::train}))
                {
                        log_error() << "invalid batch training samples!";
                }
                if (!check_samples(train_urand_samples, {f, protocol::train}))
                {
                        log_error() << "invalid random uniform training samples!";
                }

                ncv::print(train_header + " batch", train_batch_samples);
                ncv::print(train_header + " urand", train_urand_samples);

                if (!check_samples(test_batch_samples, {f, protocol::test}))
                {
                        log_error() << "invalid batch testing samples!";
                }
                if (!check_samples(test_urand_samples, {f, protocol::test}))
                {
                        log_error() << "invalid random uniform testing samples!";
                }

                ncv::print(test_header + " batch", test_batch_samples);
                ncv::print(test_header + " urand", test_urand_samples);
        }

        // OK
        log_info() << done;
        return EXIT_SUCCESS;
}
