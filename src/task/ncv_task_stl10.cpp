#include "ncv_task_stl10.h"
#include "ncv_loss.h"
#include "ncv_color.h"
#include <fstream>

namespace ncv
{
        //-------------------------------------------------------------------------------------------------

        static const string_t labels[] =
        {
                "airplane",
                "bird",
                "car",
                "cat",
                "deer",
                "dog",
                "horse",
                "monkey",
                "ship",
                "truck"
        };

        //-------------------------------------------------------------------------------------------------

        bool stl10_task::load(const string_t& dir)
        {
//                stl10/stl10_binary/class_names.txt
//                stl10/stl10_binary/fold_indices.txt

                const string_t test_ifile = dir + "/test_X.bin";
                const string_t test_gfile = dir + "/test_y.bin";
                const size_t test_n_samples = 10 * 800;

                const string_t train_ifile = dir + "/train_X.bin";
                const string_t train_gfile = dir + "/train_y.bin";
                const string_t train_uifile = dir + "/unlabeled_X.bin";
                const size_t train_n_samples = 10 * 500 + 100000;

                // TODO: folding part!

                m_images.clear();

                return  load(train_ifile, train_gfile, protocol::train) +
                        load(train_uifile, protocol::train) == train_n_samples &&

                        load(test_ifile, test_gfile, protocol::test) == test_n_samples;
        }

        //-------------------------------------------------------------------------------------------------

        size_t stl10_task::fold_size(index_t f, protocol p) const
        {

        }

        //-------------------------------------------------------------------------------------------------

        bool stl10_task::fold_sample(index_t f, protocol p, index_t s, sample& ss) const
        {

        }

        //-------------------------------------------------------------------------------------------------
        
        size_t stl10_task::load(const string_t& ifile, const string_t& gfile, protocol p)
        {
                std::ifstream fimage(ifile.c_str(), std::ios::in | std::ios::binary);
                std::ifstream flabel(gfile.c_str(), std::ios::in | std::ios::binary);                
                if (!fimage.is_open() || !flabel.is_open())
                {
                        return false;
                }

                char buffer[n_inputs()];
                char label[1];
                
                // load images and annotations
                size_t cnt = 0;
                while ( flabel.read(label, 1) &&
                        fimage.read(buffer, n_inputs()))
                {
                        const index_t ilabel = static_cast<index_t>(label[0]) - 1;
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

        size_t stl10_task::load(const string_t& ifile, protocol p)
        {
                std::ifstream fimage(ifile.c_str(), std::ios::in | std::ios::binary);
                if (!fimage.is_open())
                {
                        return false;
                }

                char buffer[n_inputs()];

                // load images
                size_t cnt = 0;
                while ( fimage.read(buffer, n_inputs()))
                {
                        annotated_image aimage;
                        aimage.m_protocol = p;
                        aimage.m_annotations.clear();
                        aimage.load_rgba(buffer, n_rows(), n_cols());

                        m_images.push_back(aimage);
                        ++ cnt;
                }

                return cnt;
        }

        //-------------------------------------------------------------------------------------------------
}
