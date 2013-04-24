#include "ncv_task_cifar10.h"
#include "ncv_random.h"
#include "ncv_color.h"
#include "ncv_loss.h"
#include <fstream>

namespace ncv
{
        //-------------------------------------------------------------------------------------------------

        cifar10_task::cifar10_task()
        {
                m_labels.push_back("airplane");
                m_labels.push_back("automobile");
                m_labels.push_back("bird");
                m_labels.push_back("cat");
                m_labels.push_back("deer");
                m_labels.push_back("dog");
                m_labels.push_back("frog");
                m_labels.push_back("horse");
                m_labels.push_back("ship");
                m_labels.push_back("truck");
        }

        //-------------------------------------------------------------------------------------------------

        cifar10_task::~cifar10_task()
        {
        }

        //-------------------------------------------------------------------------------------------------

        bool cifar10_task::load(
                const string_t& dir,
                size_t ram_gb,
                samples_t& train_samples,
                samples_t& valid_samples,
                samples_t& test_samples)
        {
                if (ram_gb < 1)
                {
                        return false;
                }

                const string_t train_bfile1 = dir + "/cifar-10-batches-bin/data_batch_1.bin";
                const string_t train_bfile2 = dir + "/cifar-10-batches-bin/data_batch_2.bin";
                const string_t train_bfile3 = dir + "/cifar-10-batches-bin/data_batch_3.bin";
                const string_t train_bfile4 = dir + "/cifar-10-batches-bin/data_batch_4.bin";
                const string_t train_bfile5 = dir + "/cifar-10-batches-bin/data_batch_5.bin";
                const size_t train_n_samples = 50000;

                const string_t test_bfile = dir + "/cifar-10-batches-bin/test_batch.bin";
                const size_t test_n_samples = 10000;

                // clear
                train_samples.clear();
                valid_samples.clear();
                test_samples.clear();

                // (50000 + 10000) *
                //      (32 * 32 * sizeof(scalar_t) +
                //            10 * sizeof(scalar_t) +
                //             1 * sizeof(scalar_t)) < 1GB

                // TODO: set costs!
                return  load(train_bfile1, train_samples, valid_samples, 0.84) &&
                        load(train_bfile2, train_samples, valid_samples, 0.84) &&
                        load(train_bfile3, train_samples, valid_samples, 0.84) &&
                        load(train_bfile4, train_samples, valid_samples, 0.84) &&
                        load(train_bfile5, train_samples, valid_samples, 0.84) &&
                        load(test_bfile, test_samples, test_samples, 2.0) &&

                        train_samples.size() + valid_samples.size() == train_n_samples &&
                        test_samples.size() == test_n_samples;
        }

        //-------------------------------------------------------------------------------------------------

        bool cifar10_task::load(const string_t& bfile,
                samples_t& samples1, samples_t& samples2, scalar_t prob)
        {
                static const index_t n_inputs = n_rows() * n_cols();

                char buffer[4096];
                char label[1];

                random<scalar_t> rgen(0.0, 1.0);
                
                // image and label data streams
                std::ifstream istream(bfile.c_str(), std::ios::in | std::ios::binary);
                if (!istream.is_open())
                {
                        return false;
                }

                // load images
                while ( istream.read(label, 1) &&
                        istream.read(buffer, 3 * n_inputs))
                {
                        // setup label
                        const index_t ilabel = static_cast<index_t>(label[0]);
                        if (ilabel >= n_labels())
                        {
                                continue;
                        }

                        // setup sample
                        sample s;
                        s.m_label = m_labels[ilabel];
                        s.m_weight = 1.0;
                        s.m_input.resize(n_rows(), n_cols());
                        s.m_target = ncv::class_target(ilabel, n_labels());

                        for (size_t y = 0, dr = 0, dg = dr + n_inputs, db = dg + n_inputs; y < n_rows(); y ++)
                        {
                                for (size_t x = 0; x < n_cols(); x ++, dr ++, dg ++, db ++)
                                {
                                        const rgba_t rgba = color::make_rgba(buffer[dr], buffer[dg], buffer[db]);
                                        const cielab_t cielab = color::make_cielab(rgba);
                                        s.m_input(y, x) = cielab(0);
                                }
                        }

                        // OK, add sample
                        (rgen() < prob ? samples1 : samples2).push_back(s);
                }

                // OK
                return true;
        }

        //-------------------------------------------------------------------------------------------------
}
