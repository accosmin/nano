#include "ncv_task_cifar10.h"
#include "ncv_random.h"
#include <fstream>

namespace ncv
{
//        //-------------------------------------------------------------------------------------------------

//        cifar10_task_t::cifar10_task_t()
//                :       class_task_t(32, 32, 10)
//        {
//                strings_t labels;
//                labels.push_back("airplane");
//                labels.push_back("automobile");
//                labels.push_back("bird");
//                labels.push_back("cat");
//                labels.push_back("deer");
//                labels.push_back("dog");
//                labels.push_back("frog");
//                labels.push_back("horse");
//                labels.push_back("ship");
//                labels.push_back("truck");

//                set_labels(labels);
//        }

//        //-------------------------------------------------------------------------------------------------

//        bool cifar10_task_t::load(const string_t& basename, const string_t& bfile,
//                data_enum dtype1, data_enum dtype2, scalar_t prob)
//        {
//                random_t<scalar_t> rgen(0.0, 1.0);
                
//                // Image and label data streams
//                std::ifstream istream(bfile.c_str(), std::ios::in | std::ios::binary);
//                if (!istream.is_open())
//                {
//                        return false;
//                }

//                // Create buffers
//                static const int BUF_SIZE = 4096;
//                char buffer[BUF_SIZE];
//                char label[1];

//                image_t image(0, 255);
//                channel_t reds(n_cols(), n_rows());
//                channel_t greens(n_cols(), n_rows());
//                channel_t blues(n_cols(), n_rows());
//                channel_t grays(n_cols(), n_rows());

//                // Now cycle over all images in the dataset
//                int cnt = 0;
//                while (istream.read(label, 1) && istream.read(buffer, 3 * n_inputs()))
//                {
//                        ++ cnt;

//                        // Setup label
//                        const index_t ilabel = cast<index_t>(label[0]);
//                        if (ilabel >= n_labels())
//                        {
//                                continue;
//                        }

//                        // Setup image
//                        for (size_t y = 0, dr = 0, dg = dr + n_inputs(), db = dg + n_inputs(); y < n_rows(); y ++)
//                        {
//                                for (size_t x = 0; x < n_cols(); x ++, dr ++, dg ++, db ++)
//                                {
//                                        reds(y, x) = cast<pixel_t>(buffer[dr]);
//                                        greens(y, x) = cast<pixel_t>(buffer[dg]);
//                                        blues(y, x) = cast<pixel_t>(buffer[db]);
//                                }
//                        }
//                        to_gray(reds, greens, blues, grays);
//                        image.load(grays, to_string(channel_enum::luma));
//                        image.rename(basename + "_" + to_string(cnt) + "_" + labels()[ilabel]);

//                        // OK, add image
//                        add_image(rgen() < prob ? dtype1 : dtype2, image, ilabel);
//                }

//                // OK
//                return true;
//        }
        
//        //-------------------------------------------------------------------------------------------------

//        bool cifar10_task_t::load(const string_t& dir)
//        {
//                const string_t train_bfile1 = dir + "/cifar-10-batches-bin/data_batch_1.bin";
//                const string_t train_bfile2 = dir + "/cifar-10-batches-bin/data_batch_2.bin";
//                const string_t train_bfile3 = dir + "/cifar-10-batches-bin/data_batch_3.bin";
//                const string_t train_bfile4 = dir + "/cifar-10-batches-bin/data_batch_4.bin";
//                const string_t train_bfile5 = dir + "/cifar-10-batches-bin/data_batch_5.bin";
//                const size_t train_n_samples = 50000;

//                const string_t test_bfile = dir + "/cifar-10-batches-bin/test_batch.bin";
//                const size_t test_n_samples = 10000;

//                clear();
                
//                return  load("train-batch1", train_bfile1, data_enum::train, data_enum::valid, 0.84) &&
//                        load("train-batch2", train_bfile2, data_enum::train, data_enum::valid, 0.84) &&
//                        load("train-batch3", train_bfile3, data_enum::train, data_enum::valid, 0.84) &&
//                        load("train-batch4", train_bfile4, data_enum::train, data_enum::valid, 0.84) &&
//                        load("train-batch5", train_bfile5, data_enum::train, data_enum::valid, 0.84) &&
//                        load("test", test_bfile, data_enum::test, data_enum::test, 2.0) &&

//                        n_images(data_enum::train) + n_images(data_enum::valid) == train_n_samples &&
//                        n_images(data_enum::test) == test_n_samples;
//        }

//        //-------------------------------------------------------------------------------------------------
}
