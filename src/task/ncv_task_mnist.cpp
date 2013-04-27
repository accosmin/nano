#include "ncv_task_mnist.h"
#include "ncv_loss.h"
#include "ncv_color.h"
#include <fstream>

namespace ncv
{
        //-------------------------------------------------------------------------------------------------

        bool mnist_task::load(const string_t& dir)
        {
                const string_t test_ifile = dir + "/t10k-images-idx3-ubyte";
                const string_t test_gfile = dir + "/t10k-labels-idx1-ubyte";
                const size_t test_n_samples = 10000;

                const string_t train_ifile = dir + "/train-images-idx3-ubyte";
                const string_t train_gfile = dir + "/train-labels-idx1-ubyte";
                const size_t train_n_samples = 60000;

                m_images.clear();

                return  load(train_ifile, train_gfile, protocol::train) == train_n_samples &&

                        load(test_ifile, test_gfile, protocol::test) == test_n_samples;
        }

        //-------------------------------------------------------------------------------------------------

        size_t mnist_task::fold_size(index_t /*f*/, protocol p) const
        {
                switch (p)
                {
                case protocol::train:
                        return 0;
                }
        }

        //-------------------------------------------------------------------------------------------------

        bool mnist_task::fold_sample(index_t /*f*/, protocol p, index_t s, sample& ss) const
        {

        }

        //-------------------------------------------------------------------------------------------------

        size_t mnist_task::load(const string_t& ifile, const string_t& gfile, protocol p)
        {
                char buffer[n_inputs()];
                char label[2];

                // image and label data streams
                std::ifstream fimage(ifile.c_str(), std::ios::in | std::ios::binary);
                std::ifstream flabel(gfile.c_str(), std::ios::in | std::ios::binary);

                if (!fimage.is_open() || !flabel.is_open())
                {
                        return 0;
                }

                // read headers
                fimage.read(buffer, 16);
                flabel.read(buffer, 8);

                // load annotations and images
                size_t cnt = 0;
                while ( flabel.read(label, 1) &&
                        fimage.read(buffer, n_inputs()))
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
                                        "digit" + text::to_string(ilabel),
                                        ncv::class_target(ilabel, n_outputs()));

                        annotated_image aimage;
                        aimage.m_protocol = p;
                        aimage.m_annotations.push_back(anno);
                        aimage.load_gray(buffer, n_rows(), n_cols());

                        m_images.push_back(aimage);
                        ++ cnt;
                }

                return cnt;
        }

        //-------------------------------------------------------------------------------------------------
}
