#include "ncv_task_cifar10.h"
#include "ncv_color.h"
#include "ncv_loss.h"
#include <fstream>

namespace ncv
{
        //-------------------------------------------------------------------------------------------------

        static const string_t labels[] =
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

        //-------------------------------------------------------------------------------------------------

        bool cifar10_task::load(const string_t& dir)
        {
                const string_t train_bfile1 = dir + "/cifar-10-batches-bin/data_batch_1.bin";
                const string_t train_bfile2 = dir + "/cifar-10-batches-bin/data_batch_2.bin";
                const string_t train_bfile3 = dir + "/cifar-10-batches-bin/data_batch_3.bin";
                const string_t train_bfile4 = dir + "/cifar-10-batches-bin/data_batch_4.bin";
                const string_t train_bfile5 = dir + "/cifar-10-batches-bin/data_batch_5.bin";
                const size_t train_n_samples = 50000;

                const string_t test_bfile = dir + "/cifar-10-batches-bin/test_batch.bin";
                const size_t test_n_samples = 10000;

                m_images.clear();

                return  load(train_bfile1, protocol::train) +
                        load(train_bfile2, protocol::train) +
                        load(train_bfile3, protocol::train) +
                        load(train_bfile4, protocol::train) +
                        load(train_bfile5, protocol::train) == train_n_samples &&

                        load(test_bfile, protocol::test) == test_n_samples;
        }

        //-------------------------------------------------------------------------------------------------

        size_t cifar10_task::fold_size(index_t /*f*/, protocol p) const
        {
                switch (p)
                {
                case protocol::train:
                        return 0;
                }
        }

        //-------------------------------------------------------------------------------------------------

        bool cifar10_task::fold_sample(index_t /*f*/, protocol p, index_t s, sample& ss) const
        {

        }

        //-------------------------------------------------------------------------------------------------

        size_t cifar10_task::load(const string_t& bfile, protocol p)
        {
                std::ifstream istream(bfile.c_str(), std::ios::in | std::ios::binary);
                if (!istream.is_open())
                {
                        return 0;
                }

                char buffer[n_inputs()];
                char label[1];

                // load images and annotations
                size_t cnt = 0;
                while ( istream.read(label, 1) &&
                        istream.read(buffer, n_inputs()))
                {
                        const index_t ilabel = static_cast<index_t>(label[0]);
                        if (ilabel >= n_outputs())
                        {
                                continue;
                        }

                        annotation anno(static_cast<coord_t>(0),
                                        static_cast<coord_t>(0),
                                        static_cast<coord_t>(n_cols()),
                                        static_cast<coord_t>(n_rows()),
                                        labels[ilabel],
                                        ncv::class_target(ilabel, n_outputs()));

                        annotated_image aimage;
                        aimage.m_protocol = p;
                        aimage.m_annotations.push_back(anno);
                        aimage.load_rgba(buffer, n_rows(), n_cols());

                        m_images.push_back(aimage);
                        ++ cnt;
                }

                return cnt;
        }

        //-------------------------------------------------------------------------------------------------
}
