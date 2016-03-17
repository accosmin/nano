#include "task_mnist.h"
#include "archive.h"
#include "io/archive.h"
#include "io/imstream.h"
#include "math/cast.hpp"
#include "cortex/class.h"
#include "text/to_string.hpp"
#include "cortex/util/logger.h"

namespace nano
{
        mnist_task_t::mnist_task_t(const string_t& configuration) :
                mem_vision_task_t("mnist", 1, 28, 28, 10, 1)
        {
        }

        bool mnist_task_t::populate(const string_t& dir)
        {
                const string_t test_ifile = dir + "/t10k-images-idx3-ubyte.gz";
                const string_t test_gfile = dir + "/t10k-labels-idx1-ubyte.gz";
                const size_t n_test_samples = 10000;

                const string_t train_ifile = dir + "/train-images-idx3-ubyte.gz";
                const string_t train_gfile = dir + "/train-labels-idx1-ubyte.gz";
                const size_t n_train_samples = 60000;

                clear_memory(n_train_samples + n_test_samples);

                return  load_binary(train_ifile, train_gfile, protocol::train, n_train_samples) &&
                        load_binary(test_ifile, test_gfile, protocol::test, n_test_samples);
        }

        bool mnist_task_t::load_binary(const string_t& ifile, const string_t& gfile,
                const protocol p, const size_t count)
        {
                size_t iindex = n_images();
                size_t icount = 0;
                size_t gcount = 0;

                const auto buffer_size = irows() * icols();
                std::vector<char> buffer = nano::make_buffer(buffer_size);
                char label[2];

                const auto error_op = [&] (const string_t& message)
                {
                        log_error() << "MNIST: " << message;
                };

                // load images
                const auto iop = [&] (const string_t&, const nano::buffer_t& data)
                {
                        nano::imstream_t stream(data.data(), data.size());

                        stream.read(buffer.data(), 16);
                        while (stream.read(buffer.data(), buffer_size))
                        {
                                image_t image;
                                image.load_luma(buffer.data(), irows(), icols());
                                add_image(image);

                                ++ icount;
                        }

                        return true;
                };

                log_info() << "MNIST: loading file <" << ifile << "> ...";
                if (!nano::unarchive(ifile, iop, error_op))
                {
                        log_error() << "MNIST: failed to load file <" << ifile << ">!";
                        return false;
                }

                // load ground truth
                const auto gop = [&] (const string_t&, const nano::buffer_t& data)
                {
                        nano::imstream_t stream(data.data(), data.size());

                        stream.read(buffer.data(), 8);
                        while (stream.read(label, 1) && stream.gcount() == 1)
                        {
                                const tensor_index_t ilabel = nano::cast<tensor_index_t>(label[0]);
                                if (ilabel >= osize())
                                {
                                        continue;
                                }

                                sample_t sample(iindex, sample_region(0, 0));
                                sample.m_label = "digit" + nano::to_string(ilabel);
                                sample.m_target = nano::class_target(ilabel, osize());
                                sample.m_fold = { 0, p };
                                add_sample(sample);

                                ++ gcount;
                                ++ iindex;
                        }

                        return true;
                };

                log_info() << "MNIST: loading file <" << gfile << "> ...";
                if (!nano::unarchive(gfile, gop, error_op))
                {
                        log_error() << "MNIST: failed to load file <" << gfile << ">!";
                        return false;
                }

                // OK
                log_info() << "MNIST: loaded " << icount << "/" << gcount << " samples.";
                return (count == gcount) && (count == icount);
        }
}
