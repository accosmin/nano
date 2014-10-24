#include "task_cifar10.h"
#include "common/logger.h"
#include "common/cast.hpp"
#include "common/io_arch.h"
#include "common/io_stream.h"
#include "loss.h"
#include <boost/algorithm/string.hpp>

namespace ncv
{
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

        cifar10_task_t::cifar10_task_t(const string_t& configuration)
                :       task_t(configuration)
        {
        }

        bool cifar10_task_t::load(const string_t& dir)
        {
                const string_t bfile = dir + "/cifar-10-binary.tar.gz";

                const string_t train_bfile1 = "data_batch_1.bin";
                const string_t train_bfile2 = "data_batch_2.bin";
                const string_t train_bfile3 = "data_batch_3.bin";
                const string_t train_bfile4 = "data_batch_4.bin";
                const string_t train_bfile5 = "data_batch_5.bin";
                const size_t n_train_samples = 50000;

                const string_t test_bfile = "test_batch.bin";
                const size_t n_test_samples = 10000;

                clear_memory(n_train_samples + n_test_samples);

                const auto op = [&] (const string_t& filename, const io::data_t& data)
                {
                        if (    boost::algorithm::iends_with(filename, train_bfile1) ||
                                boost::algorithm::iends_with(filename, train_bfile2) ||
                                boost::algorithm::iends_with(filename, train_bfile3) ||
                                boost::algorithm::iends_with(filename, train_bfile4) ||
                                boost::algorithm::iends_with(filename, train_bfile5))
                        {
                                log_info() << "CIFAR-10: loading file <" << filename << "> ...";
                                load(data, protocol::train);
                        }

                        else if (boost::algorithm::iends_with(filename, test_bfile))
                        {
                                log_info() << "CIFAR-10: loading file <" << filename << "> ...";
                                load(data, protocol::test);
                        }
                };

                log_info() << "CIFAR-10: loading file <" << bfile << "> ...";

                return  io::decode(bfile, "CIFAR-10: ", op) &&
                        m_samples.size() == n_train_samples + n_test_samples &&
                        m_images.size() == n_train_samples + n_test_samples;
        }

        size_t cifar10_task_t::load(const io::data_t& data, protocol p)
        {
                std::vector<char> vbuffer(n_rows() * n_cols() * 3);
                char* buffer = vbuffer.data();
                char label[1];

                io::stream_t stream(data);

                size_t cnt = 0;
                while ( stream.read(label, 1) &&
                        stream.read(buffer, vbuffer.size()))
                {
                        const size_t ilabel = math::cast<size_t>(label[0]);

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
}
