#include "nanocv.h"
#include <algorithm>

using namespace ncv;

bool check_samples(const samples_t& samples, ncv::fold_t fold)
{
        return std::find_if(samples.begin(), samples.end(),
               [&] (const sample_t& sample) { return sample.m_fold != fold; }) == samples.end();
}

void print_samples(const string_t& name, const samples_t& samples, size_t n_labels)
{
        for (size_t ilabel = 0; ilabel < n_labels; ilabel ++)
        {
                const string_t label = "label" + text::to_string(ilabel);

                sampler_t sampler(samples);
                sampler.setup(sampler_t::stype::batch);
                sampler.setup(label);

                const samples_t lsamples = sampler.get();
                log_info() << name << " [" << label << "]: count = " << lsamples.size() << "/" << samples.size()
                           << ", weights = " << ncv::accumulate(lsamples) << ".";
        }
}

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
                void resize(size_t size)
                {
                        random_t<size_t> rng(1, 100);

                        m_images.clear();
                        m_samples.clear();

                        for (size_t f = 0, idx = 0; f < n_folds(); f ++)
                        {
                                for (protocol p : {protocol::train, protocol::test})
                                {
                                        for (size_t i = 0; i < size / 2; i ++)
                                        {
                                                const size_t ilabel = (rng() < 50) ? 0 : (rng() % n_outputs());

                                                sample_t sample(m_images.size(), sample_region(0, 0), ilabel == 0 ? 50.0 : 1.0);
                                                sample.m_label = "label" + text::to_string(ilabel);
                                                sample.m_target = ncv::class_target(ilabel, n_outputs());
                                                sample.m_fold = {f, p};
                                                m_samples.push_back(sample);

                                                image_t image(n_rows(), n_cols(), color());
                                                m_images.push_back(image);
                                        }

                                        // normalize weights
                                        const samples_it_t begin = m_samples.begin() + idx;
                                        const samples_it_t end = m_samples.begin() + (idx + size / 2);

                                        ncv::label_normalize(begin, end);
                                        idx += size / 2;

                                        // debug
                                        print_samples("fold [" + text::to_string(f + 1) + "/" +
                                                      text::to_string(n_folds()) + "] " +
                                                      "protocol [" + text::to_string(p) + "]",
                                                      samples_t(begin, end), n_outputs());
                                }
                        }
                }

                // load images from the given directory
                virtual bool load(const string_t&) { return true; }

                // access functions
                virtual size_t n_rows() const { return 28; }
                virtual size_t n_cols() const { return 28; }
                virtual size_t n_outputs() const { return 3; }
                virtual size_t n_folds() const { return 5; }
                virtual color_mode color() const { return color_mode::luma; }
        };
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

                const fold_t train_fold = {f, protocol::train};
                const fold_t test_fold = {f, protocol::test};

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

                // random weight-based training samples
                sampler.reset();
                sampler.setup(train_fold);
                sampler.setup(sampler_t::stype::weighted, n_rand_samples);

                const samples_t train_wrand_samples = sampler.get();

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

                // random weight-based testing samples
                sampler.reset();
                sampler.setup(test_fold);
                sampler.setup(sampler_t::stype::weighted, n_rand_samples);

                const samples_t test_wrand_samples = sampler.get();

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
                if (!check_samples(train_wrand_samples, {f, protocol::train}))
                {
                        log_error() << "invalid random weighted training samples!";
                }

                print_samples("train batch", train_batch_samples, task.n_outputs());
                print_samples("train urand", train_urand_samples, task.n_outputs());
                print_samples("train wrand", train_wrand_samples, task.n_outputs());

                if (!check_samples(test_batch_samples, {f, protocol::test}))
                {
                        log_error() << "invalid batch testing samples!";
                }
                if (!check_samples(test_urand_samples, {f, protocol::test}))
                {
                        log_error() << "invalid random uniform testing samples!";
                }
                if (!check_samples(test_wrand_samples, {f, protocol::test}))
                {
                        log_error() << "invalid random weighted testing samples!";
                }

                print_samples("test batch", test_batch_samples, task.n_outputs());
                print_samples("test urand", test_urand_samples, task.n_outputs());
                print_samples("test wrand", test_wrand_samples, task.n_outputs());
        }

        // OK
        log_info() << done;
        return EXIT_SUCCESS;
}
