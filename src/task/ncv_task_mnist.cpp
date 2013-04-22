#include "ncv_task_mnist.h"
#include "ncv_random.h"
#include <fstream>

namespace ncv
{
        //-------------------------------------------------------------------------------------------------

        mnist_task::mnist_task()
        {
                for (size_t i = 0; i < 10; i ++)
                {
                        m_labels.push_back("digit-" + text::to_string(i));
                }
        }

        //-------------------------------------------------------------------------------------------------

        mnist_task::~mnist_task()
        {
        }

        //-------------------------------------------------------------------------------------------------

        bool mnist_task::load(
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

                const string_t test_ifile = dir + "/t10k-images-idx3-ubyte";
                const string_t test_lfile = dir + "/t10k-labels-idx1-ubyte";
                const size_t test_n_samples = 10000;

                const string_t train_ifile = dir + "/train-images-idx3-ubyte";
                const string_t train_lfile = dir + "/train-labels-idx1-ubyte";
                const size_t train_n_samples = 60000;

                // clear
                train_samples.clear();
                valid_samples.clear();
                test_samples.clear();

                // (60000 + 10000) *
                //      (28 * 28 * sizeof(scalar_t) +
                //            10 * sizeof(scalar_t) +
                //             1 * sizeof(scalar_t)) < 1GB

                return  load("train", train_ifile, train_lfile, train_samples, valid_samples, 0.84) &&
                        load("test", test_ifile, test_lfile, test_samples, test_samples, 2.0) &&

                        train_samples.size() + valid_samples.size() == train_n_samples &&
                        test_samples.size() == test_n_samples;

                // TODO: set costs!

                return true;
        }

        //-------------------------------------------------------------------------------------------------
        
        bool mnist_task::load(const string_t& basename, const string_t& ifile, const string_t& gfile,
                samples_t& samples1, samples_t& samples2, scalar_t prob)
        {
                random<scalar_t> rgen(0.0, 1.0);
                
                // Image and label data streams
                std::ifstream fimage(ifile.c_str(), std::ios::in | std::ios::binary);
                std::ifstream flabel(gfile.c_str(), std::ios::in | std::ios::binary);
                
                if (!fimage.is_open() || !flabel.is_open())
                {
                        return false;
                }
                
                // Create buffers
                static const int BUF_SIZE = 2048;
                char buffer[BUF_SIZE];
                char label[2];

                image_t image(0, 255);
                channel_t grays(n_cols(), n_rows());
        
                // Read headers
                fimage.read(buffer, 16);
                flabel.read(buffer, 8);
                
                // Load the images in the dataset
                int cnt = 0;
                while (flabel.read(label, 1) && fimage.read(buffer, n_inputs()))
                {
                        ++ cnt;

                        // Setup label
                        const index_t ilabel = cast<index_t>(label[0]);
                        if (ilabel >= n_labels())
                        {
                                continue;
                        }

                        // Setup image
                        for (index_t y = 0, i = 0; y < n_rows(); y ++)
                        {
                                for (index_t x = 0; x < n_cols(); x ++, i ++)
                                {
                                        grays(y, x) = cast<pixel_t>(buffer[i]);
                                }
                        }
                        image.load(grays, to_string(channel_enum::luma));
                        image.rename(basename + "_" + to_string(cnt) + "_" + labels()[ilabel]);

                        // OK, add image
                        add_image(rgen() < prob ? dtype1 : dtype2, image, ilabel);
                }
                
                // OK
                return true;
        }

        //-------------------------------------------------------------------------------------------------
}
