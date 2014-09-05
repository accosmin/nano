#include "nanocv.h"
#include <algorithm>

namespace ncv
{
        class dummy_tast_t : public ncv::task_t
        {
        public:

                NANOCV_MAKE_CLONABLE(dummy_tast_t)

                // constructor
                dummy_tast_t(const string_t& = string_t())
                        :       task_t("test task")
                {
                }

                // create samples
                void resize(size_t samples)
                {
                        m_images.clear();
                        m_samples.clear();

                        for (size_t f = 0; f < n_folds(); f ++)
                        {
                                for (size_t i = 0; i < samples; i ++)
                                {
                                        sample_t sample(m_images.size(), sample_region(0, 0));
                                        sample.m_label = "label";
                                        sample.m_target = ncv::class_target(i % n_outputs(), n_outputs());
                                        sample.m_fold = { f, (i % 2 == 0) ? protocol::train : protocol::test };
                                        m_samples.push_back(sample);

                                        image_t image(n_rows(), n_cols(), color());
                                        m_images.push_back(image);
                                }
                        }
                }

                // load images from the given directory
                virtual bool load(const string_t&) { return true; }

                // access functions
                virtual size_t n_rows() const { return 28; }
                virtual size_t n_cols() const { return 28; }
                virtual size_t n_outputs() const { return 10; }
                virtual size_t n_folds() const { return 10; }
                virtual color_mode color() const { return color_mode::luma; }
        };
}

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

        dummy_tast_t task;
        task.resize(n_samples);

        for (size_t f = 0; f < task.n_folds(); f ++)
        {
                const ncv::timer_t timer;

                // batch training samples
                sampler_t sampler(task);
                sampler.reset();
                sampler.setup(fold_t{ f, protocol::train });
                sampler.setup(sampler_t::stype::batch);

                const samples_t train_batch_samples = sampler.get();

                // random training samples
                sampler.reset();
                sampler.setup(fold_t{ f, protocol::train });
                sampler.setup(sampler_t::stype::uniform, n_rand_samples);

                const samples_t train_rand_samples = sampler.get();

                // batch testing samples
                sampler.reset();
                sampler.setup(fold_t{ f, protocol::test });
                sampler.setup(sampler_t::stype::batch);

                const samples_t test_batch_samples = sampler.get();

                // random testing samples
                sampler.reset();
                sampler.setup(fold_t{ f, protocol::test });
                sampler.setup(sampler_t::stype::uniform, n_rand_samples);

                const samples_t test_rand_samples = sampler.get();

                log_info() << "fold [" << (f + 1) << "/" << task.n_folds() << "]: sampled in " << timer.elapsed() << ".";

                // check results
                if (train_batch_samples.size() + test_batch_samples.size() != n_samples)
                {
                        log_error() << "invalid number of batch samples "
                                    << (train_batch_samples.size() + test_batch_samples.size())
                                    << "/" << n_samples << "!";
                }

                if (!check_samples(train_batch_samples, fold_t{ f, protocol::train}))
                {
                        log_error() << "invalid batch training samples!";
                }
                if (!check_samples(train_rand_samples, fold_t{ f, protocol::train}))
                {
                        log_error() << "invalid random training samples!";
                }

                if (!check_samples(test_batch_samples, fold_t{ f, protocol::test}))
                {
                        log_error() << "invalid batch testing samples!";
                }
                if (!check_samples(test_rand_samples, fold_t{ f, protocol::test}))
                {
                        log_error() << "invalid random testing samples!";
                }
        }

        // OK
        log_info() << done;
        return EXIT_SUCCESS;
}
