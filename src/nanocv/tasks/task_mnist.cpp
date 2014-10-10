#include "task_mnist.h"
#include "common/logger.h"
#include "common/math.hpp"
#include "common/cast.hpp"
#include "loss.h"
#include <fstream>

namespace ncv
{
        mnist_task_t::mnist_task_t(const string_t& configuration)
                :       task_t(configuration)
        {
        }

        bool mnist_task_t::load(const string_t& dir)
        {
                const string_t test_ifile = dir + "/t10k-images-idx3-ubyte";
                const string_t test_gfile = dir + "/t10k-labels-idx1-ubyte";
                const size_t n_test_samples = 10000;

                const string_t train_ifile = dir + "/train-images-idx3-ubyte";
                const string_t train_gfile = dir + "/train-labels-idx1-ubyte";
                const size_t n_train_samples = 60000;

                clear_memory(n_train_samples + n_test_samples);

                return  load(train_ifile, train_gfile, protocol::train) == n_train_samples &&
                        load(test_ifile, test_gfile, protocol::test) == n_test_samples;
        }

        size_t mnist_task_t::load(const string_t& ifile, const string_t& gfile, protocol p)
        {
                std::vector<char> vbuffer(n_rows() * n_cols());
                char* buffer = vbuffer.data();
                char label[2];

                log_info() << "MNIST: loading file <" << ifile << "> ...";

                // image and label data streams
                std::ifstream fimage(ifile.c_str(), std::ios::in | std::ios::binary);
                std::ifstream flabel(gfile.c_str(), std::ios::in | std::ios::binary);

                if (!fimage.is_open() || !flabel.is_open())
                {
                        log_error() << "MNIST: failed to load files!";
                        return 0;
                }

                // read headers
                fimage.read(buffer, 16);
                flabel.read(buffer, 8);

                // load annotations and images
                size_t cnt = 0;
                while ( flabel.read(label, 1) &&
                        fimage.read(buffer, vbuffer.size()))
                {
                        const size_t ilabel = math::cast<size_t>(label[0]);
                        if (ilabel >= n_outputs())
                        {
                                continue;
                        }

                        sample_t sample(m_images.size(), sample_region(0, 0));
                        sample.m_label = "digit" + text::to_string(ilabel);
                        sample.m_target = ncv::class_target(ilabel, n_outputs());
                        sample.m_fold = { 0, p };
                        m_samples.push_back(sample);

                        image_t image;
                        image.load_luma(buffer, n_rows(), n_cols());
                        m_images.push_back(image);

                        ++ cnt;
                }

                log_info() << "MNIST: loaded " << cnt << " samples.";

                return cnt;
        }
}
