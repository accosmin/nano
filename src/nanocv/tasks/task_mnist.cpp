#include "task_mnist.h"
#include "common/logger.h"
#include "common/math.hpp"
#include "common/cast.hpp"
#include "common/io_arch.h"
#include "common/io_stream.h"
#include "loss.h"

namespace ncv
{
        mnist_task_t::mnist_task_t(const string_t& configuration)
                :       task_t(configuration)
        {
        }

        bool mnist_task_t::load(const string_t& dir)
        {
                const string_t test_ifile = dir + "/t10k-images-idx3-ubyte.gz";
                const string_t test_gfile = dir + "/t10k-labels-idx1-ubyte.gz";
                const size_t n_test_samples = 10000;

                const string_t train_ifile = dir + "/train-images-idx3-ubyte.gz";
                const string_t train_gfile = dir + "/train-labels-idx1-ubyte.gz";
                const size_t n_train_samples = 60000;

                clear_memory(n_train_samples + n_test_samples);

                return  load(train_ifile, train_gfile, protocol::train) == n_train_samples &&
                        load(test_ifile, test_gfile, protocol::test) == n_test_samples;
        }

        size_t mnist_task_t::load(const string_t& ifile, const string_t& gfile, protocol p)
        {
                size_t icnt = 0, isize = m_images.size();
                size_t gcnt = 0;

                std::vector<char> vbuffer(n_rows() * n_cols());
                char* buffer = vbuffer.data();
                char label[2];

                // load image file
                const auto iop = [&] (const string_t&, const io::data_t& data)
                {
                        io::stream_t stream(data);

                        stream.read(buffer, 16);
                        while (stream.read(buffer, vbuffer.size()) && stream.gcount() == vbuffer.size())
                        {
                                image_t image;
                                image.load_luma(buffer, n_rows(), n_cols());
                                m_images.push_back(image);

                                ++ icnt;
                        }
                };

                log_info() << "MNIST: loading file <" << ifile << "> ...";
                if (!io::decode(ifile, "MNIST: ", iop))
                {
                        log_error() << "MNIST: failed to load file <" << ifile << ">!";
                        return 0;
                }

                // load ground truth file
                const auto gop = [&] (const string_t&, const io::data_t& data)
                {
                        io::stream_t stream(data);

                        stream.read(buffer, 8);
                        while (stream.read(label, 1) && stream.gcount() == 1)
                        {
                                const size_t ilabel = math::cast<size_t>(label[0]);

                                sample_t sample(isize, sample_region(0, 0));
                                sample.m_label = "digit" + text::to_string(ilabel);
                                sample.m_target = ncv::class_target(ilabel, n_outputs());
                                sample.m_fold = { 0, p };
                                m_samples.push_back(sample);

                                ++ gcnt;
                                ++ isize;
                        }
                };

                log_info() << "MNIST: loading file <" << gfile << "> ...";
                if (!io::decode(gfile, "MNIST: ", gop))
                {
                        log_error() << "MNIST: failed to load file <" << gfile << ">!";
                        return 0;
                }

                // OK
                log_info() << "MNIST: loaded " << icnt << "/" << gcnt << " samples.";
                return std::max(icnt, gcnt);
        }
}
