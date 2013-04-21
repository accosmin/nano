#include "ncv_task_mnist.h"
#include "ncv_random.h"
#include <fstream>

namespace ncv
{
//        //-------------------------------------------------------------------------------------------------

//        mnist_task_t::mnist_task_t()
//                :       class_task_t(28, 28, 10)
//        {
//                strings_t labels;
//                for (size_t i = 0; i < n_outputs(); i ++)
//                {
//                        labels.push_back("digit-" + to_string(i));
//                }

//                set_labels(labels);
//        }

//        //-------------------------------------------------------------------------------------------------
        
//        bool mnist_task_t::load(const string_t& basename, const string_t& ifile, const string_t& gfile,
//                data_enum dtype1, data_enum dtype2, scalar_t prob)
//        {
//                random_t<scalar_t> rgen(0.0, 1.0);
                
//                // Image and label data streams
//                std::ifstream fimage(ifile.c_str(), std::ios::in | std::ios::binary);
//                std::ifstream flabel(gfile.c_str(), std::ios::in | std::ios::binary);
                
//                if (!fimage.is_open() || !flabel.is_open())
//                {
//                        return false;
//                }
                
//                // Create buffers
//                static const int BUF_SIZE = 2048;
//                char buffer[BUF_SIZE];
//                char label[2];

//                image_t image(0, 255);
//                channel_t grays(n_cols(), n_rows());
        
//                // Read headers
//                fimage.read(buffer, 16);
//                flabel.read(buffer, 8);
                
//                // Load the images in the dataset
//                int cnt = 0;
//                while (flabel.read(label, 1) && fimage.read(buffer, n_inputs()))
//                {
//                        ++ cnt;

//                        // Setup label
//                        const index_t ilabel = cast<index_t>(label[0]);
//                        if (ilabel >= n_labels())
//                        {
//                                continue;
//                        }

//                        // Setup image
//                        for (index_t y = 0, i = 0; y < n_rows(); y ++)
//                        {
//                                for (index_t x = 0; x < n_cols(); x ++, i ++)
//                                {
//                                        grays(y, x) = cast<pixel_t>(buffer[i]);
//                                }
//                        }
//                        image.load(grays, to_string(channel_enum::luma));
//                        image.rename(basename + "_" + to_string(cnt) + "_" + labels()[ilabel]);

//                        // OK, add image
//                        add_image(rgen() < prob ? dtype1 : dtype2, image, ilabel);
//                }
                
//                // OK
//                return true;
//        }
        
//        //-------------------------------------------------------------------------------------------------

//        bool mnist_task_t::load(const string_t& dir)
//        {
//                const string_t test_ifile = dir + "/t10k-images-idx3-ubyte";
//                const string_t test_lfile = dir + "/t10k-labels-idx1-ubyte";
//                const size_t test_n_samples = 10000;
                
//                const string_t train_ifile = dir + "/train-images-idx3-ubyte";
//                const string_t train_lfile = dir + "/train-labels-idx1-ubyte";
//                const size_t train_n_samples = 60000;

//                clear();
                
//                return  load("train", train_ifile, train_lfile, data_enum::train, data_enum::valid, 0.84) &&
//                        load("test", test_ifile, test_lfile, data_enum::test, data_enum::test, 2.0) &&

//                        n_images(data_enum::train) + n_images(data_enum::valid) == train_n_samples &&
//                        n_images(data_enum::test) == test_n_samples;
//        }

//        //-------------------------------------------------------------------------------------------------
}
