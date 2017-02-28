#include "class.h"
#include "logger.h"
#include "task_mnist.h"
#include "io/archive.h"
#include "text/to_params.h"
#include "text/from_params.h"

namespace nano
{
        static const string_t tlabels[] =
        {
                "digit0",
                "digit1",
                "digit2",
                "digit3",
                "digit4",
                "digit5",
                "digit6",
                "digit7",
                "digit8",
                "digit9"
        };

        mnist_task_t::mnist_task_t(const string_t& config) :
                mem_vision_task_t(dim3d_t{1, 28, 28}, dim3d_t{10, 1, 1}, 1,
                to_params(config, "dir", string_t(std::getenv("HOME")) + "/experiments/databases/mnist"))
        {
        }

        bool mnist_task_t::populate()
        {
                const string_t dir = from_params<string_t>(config(), "dir");

                const string_t test_ifile = dir + "/t10k-images-idx3-ubyte.gz";
                const string_t test_gfile = dir + "/t10k-labels-idx1-ubyte.gz";
                const size_t n_test_samples = 10000;

                const string_t train_ifile = dir + "/train-images-idx3-ubyte.gz";
                const string_t train_gfile = dir + "/train-labels-idx1-ubyte.gz";
                const size_t n_train_samples = 60000;

                return  load_binary(train_ifile, train_gfile, protocol::train, n_train_samples) &&
                        load_binary(test_ifile, test_gfile, protocol::test, n_test_samples);
        }

        bool mnist_task_t::load_binary(const string_t& ifile, const string_t& gfile,
                const protocol p, const size_t count)
        {
                size_t iindex = n_chunks();
                size_t icount = 0;
                size_t gcount = 0;

                const auto irows = std::get<1>(idims());
                const auto icols = std::get<2>(idims());
                const auto buffer_size = irows * icols;

                std::vector<char> buffer(static_cast<size_t>(buffer_size));
                char label[2];

                const auto error_op = [&] (const string_t& message)
                {
                        log_error() << "MNIST: " << message;
                };

                // load images
                const auto iop = [&] (const string_t&, istream_t& stream)
                {
                        if (stream.read(buffer.data(), 16) != 16)
                        {
                                return false;
                        }

                        while (stream.read(buffer.data(), buffer_size) == buffer_size)
                        {
                                image_t image;
                                image.load_luma(buffer.data(), irows, icols);
                                add_chunk(image, image.hash());

                                ++ icount;
                        }

                        return true;
                };

                log_info() << "MNIST: loading file <" << ifile << "> ...";
                if (!nano::load_archive(ifile, iop, error_op))
                {
                        log_error() << "MNIST: failed to load file <" << ifile << ">!";
                        return false;
                }

                // load ground truth
                const auto gop = [&] (const string_t&, istream_t& stream)
                {
                        if (stream.read(buffer.data(), 8) != 8)
                        {
                                return false;
                        }

                        while (stream.read(label, 1) == 1)
                        {
                                const tensor_index_t ilabel = static_cast<tensor_index_t>(label[0]);
                                if (ilabel < 0 || ilabel >= nano::size(odims()))
                                {
                                        log_error() << "MNIST: invalid label!";
                                        return false;
                                }

                                const auto fold = make_fold(0, p);
                                add_sample(fold, iindex, class_target(ilabel, nano::size(odims())), tlabels[ilabel]);

                                ++ gcount;
                                ++ iindex;
                        }

                        return true;
                };

                log_info() << "MNIST: loading file <" << gfile << "> ...";
                if (!nano::load_archive(gfile, gop, error_op))
                {
                        log_error() << "MNIST: failed to load file <" << gfile << ">!";
                        return false;
                }

                // OK
                log_info() << "MNIST: loaded " << icount << "/" << gcount << " samples.";
                return (count == gcount) && (count == icount);
        }
}
