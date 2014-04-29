#include "task_cifar10.h"
#include "common/logger.h"
#include "common/math.hpp"
#include "loss.h"
#include <fstream>

namespace ncv
{
        /////////////////////////////////////////////////////////////////////////////////////////

        static const string_t tlabels[] =
        {
                "airplane",
                "automobile",
                "bird",
                "cat",
                "deer",
                "dog",
                "frog",
                "horse",
                "ship",
                "truck"
        };

        /////////////////////////////////////////////////////////////////////////////////////////

        bool cifar10_task_t::load(const string_t& dir)
        {
                const string_t train_bfile1 = dir + "/data_batch_1.bin";
                const string_t train_bfile2 = dir + "/data_batch_2.bin";
                const string_t train_bfile3 = dir + "/data_batch_3.bin";
                const string_t train_bfile4 = dir + "/data_batch_4.bin";
                const string_t train_bfile5 = dir + "/data_batch_5.bin";
                const size_t n_train_images = 50000;

                const string_t test_bfile = dir + "/test_batch.bin";
                const size_t n_test_images = 10000;

                m_images.clear();
                m_samples.clear();

                return  load(train_bfile1, protocol::train) +
                        load(train_bfile2, protocol::train) +
                        load(train_bfile3, protocol::train) +
                        load(train_bfile4, protocol::train) +
                        load(train_bfile5, protocol::train) == n_train_images &&
                        load(test_bfile, protocol::test) == n_test_images;
        }

        /////////////////////////////////////////////////////////////////////////////////////////

        size_t cifar10_task_t::load(const string_t& bfile, protocol p)
        {
                log_info() << "CIFAR-10: loading file <" << bfile << "> ...";

                std::ifstream istream(bfile.c_str(), std::ios::in | std::ios::binary);
                if (!istream.is_open())
                {
                        log_error() << "CIFAR-10: failed to open file!";
                        return 0;
                }

                std::vector<char> vbuffer(n_rows() * n_cols() * 3);
                char* buffer = vbuffer.data();
                char label[1];

                // load images and annotations
                size_t cnt = 0;
                while ( istream.read(label, 1) &&
                        istream.read(buffer, vbuffer.size()))
                {
                        const size_t ilabel = math::cast<size_t>(label[0]);
                        if (ilabel >= n_outputs())
                        {
                                continue;
                        }

                        sample_t sample(m_images.size(), sample_region(0, 0));
                        sample.m_label = tlabels[ilabel];
                        sample.m_target = ncv::class_target(ilabel, n_outputs());
                        sample.m_fold = { 0, p };
                        m_samples.push_back(sample);

                        image_t image;
                        image.load_rgba(buffer, n_rows(), n_cols(), n_rows() * n_cols());
                        m_images.push_back(image);

                        ++ cnt;
                }

                log_info() << "CIFAR-10: loaded " << cnt << " samples.";

                return cnt;
        }

        /////////////////////////////////////////////////////////////////////////////////////////
}
