#include "archive.h"
#include "io/archive.h"
#include "io/imstream.h"
#include "math/cast.hpp"
#include "task_cifar10.h"
#include "cortex/class.h"
#include "text/algorithm.h"
#include "cortex/util/logger.h"

namespace nano
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

        cifar10_task_t::cifar10_task_t(const string_t&) :
                mem_vision_task_t("cifar-10", 3, 32, 32, 10)
        {
        }

        bool cifar10_task_t::populate(const string_t& dir)
        {
                const string_t bfile = dir + "/cifar-10-binary.tar.gz";

                const string_t train_bfile1 = "data_batch_1.bin";
                const string_t train_bfile2 = "data_batch_2.bin";
                const string_t train_bfile3 = "data_batch_3.bin";
                const string_t train_bfile4 = "data_batch_4.bin";
                const string_t train_bfile5 = "data_batch_5.bin";
                const size_t n_train_samples = 10000;// * 5

                const string_t test_bfile = "test_batch.bin";
                const size_t n_test_samples = 10000;

                const auto op = [&] (const string_t& filename, const nano::buffer_t& data)
                {
                        if (    nano::iends_with(filename, train_bfile1) ||
                                nano::iends_with(filename, train_bfile2) ||
                                nano::iends_with(filename, train_bfile3) ||
                                nano::iends_with(filename, train_bfile4) ||
                                nano::iends_with(filename, train_bfile5))
                        {
                                return load(filename, data.data(), data.size(), protocol::train, n_train_samples);
                        }
                        else if (nano::iends_with(filename, test_bfile))
                        {
                                return load(filename, data.data(), data.size(), protocol::test, n_test_samples);
                        }
                        else
                        {
                                return true;
                        }
                };
                const auto error_op = [&] (const string_t& message)
                {
                        log_error() << "CIFAR-10: " << message;
                };

                log_info() << "CIFAR-10: loading file <" << bfile << "> ...";

                return nano::unarchive(bfile, op, error_op);
        }

        bool cifar10_task_t::load(const string_t& filename, const char* bdata, size_t bdata_size, protocol p, size_t count)
        {
                log_info() << "CIFAR-10: loading file <" << filename << "> ...";

                const auto buffer_size = irows() * icols() * 3;
                std::vector<char> buffer = nano::make_buffer(buffer_size);
                char label[1];

                nano::imstream_t stream(bdata, bdata_size);

                size_t icount = 0;
                while ( stream.read(label, 1) &&
                        stream.read(buffer.data(), buffer_size))
                {
                        const tensor_index_t ilabel = label[0];
                        if (ilabel >= osize())
                        {
                                continue;
                        }

                        image_t image;
                        image.load_rgba(buffer.data(), irows(), icols(), irows() * icols());
                        add_image(image);

                        sample_t sample(n_images() - 1, sample_region(0, 0));
                        sample.m_label = tlabels[ilabel];
                        sample.m_target = nano::class_target(ilabel, osize());
                        sample.m_fold = { 0, p };
                        add_sample(sample);

                        ++ icount;
                }

                // OK
                log_info() << "CIFAR-10: loaded " << icount << " samples.";
                return (count == icount);
        }
}
