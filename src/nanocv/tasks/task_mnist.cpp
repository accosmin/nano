#include "task_mnist.h"
#include "common/logger.h"
#include "common/math.hpp"
#include "loss.h"
#include <fstream>

namespace ncv
{
        /////////////////////////////////////////////////////////////////////////////////////////

        bool mnist_task_t::load(const string_t& dir)
        {
                const string_t test_ifile = dir + "/t10k-images-idx3-ubyte";
                const string_t test_gfile = dir + "/t10k-labels-idx1-ubyte";
                const size_t n_test_samples = 10000;

                const string_t train_ifile = dir + "/train-images-idx3-ubyte";
                const string_t train_gfile = dir + "/train-labels-idx1-ubyte";
                const size_t n_train_samples = 60000;

                m_images.clear();
                m_folds.clear();

                return  load(train_ifile, train_gfile, protocol::train) == n_train_samples &&
                        load(test_ifile, test_gfile, protocol::test) == n_test_samples &&
                        build_folds(n_train_samples, n_test_samples);
        }

        /////////////////////////////////////////////////////////////////////////////////////////

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

                        const annotation_t anno(sample_region(0, 0),
                                "digit" + text::to_string(ilabel),
                                ncv::class_target(ilabel, n_outputs()));

                        image_t image;
                        image.m_protocol = p;
                        image.m_annotations.push_back(anno);
                        image.load_gray(buffer, n_rows(), n_cols());

                        m_images.push_back(image);
                        ++ cnt;
                }

                log_info() << "MNIST: loaded " << cnt << " samples.";

                return cnt;
        }

        /////////////////////////////////////////////////////////////////////////////////////////

        bool mnist_task_t::build_folds(size_t n_train, size_t n_test)
        {
                const fold_t train_fold = std::make_pair(0, protocol::train);
                m_folds[train_fold] = make_samples(0, n_train, sample_region(0, 0));

                const fold_t test_fold = std::make_pair(0, protocol::test);
                m_folds[test_fold] = make_samples(n_train, n_test, sample_region(0, 0));

                return true;
        }

        /////////////////////////////////////////////////////////////////////////////////////////
}
