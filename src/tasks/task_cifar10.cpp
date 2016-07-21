#include "archive.h"
#include "io/archive.h"
#include "io/imstream.h"
#include "math/cast.hpp"
#include "task_cifar10.h"
#include "class.h"
#include "logger.h"
#include "text/from_params.hpp"

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

        cifar10_task_t::cifar10_task_t(const string_t& config) :
                mem_vision_task_t("cifar-10", 3, 32, 32, 10, 1, config)
        {
        }

        bool cifar10_task_t::populate()
        {
                const string_t dir = nano::from_params<string_t>(configuration(), "dir", ".");

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
                                return load_binary(filename, data.data(), data.size(), protocol::train, n_train_samples);
                        }
                        else if (nano::iends_with(filename, test_bfile))
                        {
                                return load_binary(filename, data.data(), data.size(), protocol::test, n_test_samples);
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

        bool cifar10_task_t::load_binary(const string_t& filename,
                const char* bdata, const size_t bdata_size, const protocol p, const size_t count)
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
                        image.load_rgb(buffer.data(), irows(), icols(), irows() * icols());
                        add_chunk(image);

                        const auto fold = make_fold(0, p);
                        add_sample(fold, n_chunks() - 1, class_target(ilabel, osize()), tlabels[ilabel]);

                        ++ icount;
                }

                // OK
                log_info() << "CIFAR-10: loaded " << icount << " samples.";
                return (count == icount);
        }
}
