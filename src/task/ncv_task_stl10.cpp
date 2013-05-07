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

        bool stl10_task_t::load(const string_t& dir)
        {
                const string_t test_ifile = dir + "/test_X.bin";
                const string_t test_gfile = dir + "/test_y.bin";
                const size_t n_test_samples = 10 * 800;

                const string_t train_ifile = dir + "/train_X.bin";
                const string_t train_gfile = dir + "/train_y.bin";
                const string_t train_uifile = dir + "/unlabeled_X.bin";
                const size_t n_train_samples = 10 * 500 + 100000;

                const string_t fold_indices_file = dir + "/fold_indices.txt";

                m_images.clear();
                m_folds.clear();

                return  load(train_ifile, train_gfile, protocol::train) +
                        load(train_uifile, protocol::train) == n_train_samples &&
                        load(test_ifile, test_gfile, protocol::test) == n_test_samples &&
                        build_folds(fold_indices_file, 5000, 100000, n_test_samples);
        }

        //-------------------------------------------------------------------------------------------------
        
        size_t stl10_task_t::load(const string_t& ifile, const string_t& gfile, protocol p)
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

                        const annotation_t anno(
                                static_cast<coord_t>(0),
                                static_cast<coord_t>(0),
                                static_cast<coord_t>(n_cols()),
                                static_cast<coord_t>(n_rows()),
                                labels[ilabel],
                                ncv::class_target(ilabel, n_outputs()));

                        annotated_image_t aimage;
                        aimage.m_protocol = p;
                        aimage.m_annotations.push_back(anno);
                        aimage.load_rgba(buffer, n_rows(), n_cols());

                        m_images.push_back(aimage);
                        ++ cnt;
                }

                return cnt;
        }

        //-------------------------------------------------------------------------------------------------

        size_t stl10_task_t::load(const string_t& ifile, protocol p)
        {
                std::ifstream fimage(ifile.c_str(), std::ios::in | std::ios::binary);
                if (!fimage.is_open())
                {
                        return false;
                }

                char buffer[n_inputs()];

                // load images
                size_t cnt = 0;
                while (fimage.read(buffer, n_inputs()))
                {
                        annotated_image_t aimage;
                        aimage.m_protocol = p;
                        aimage.m_annotations.clear();
                        aimage.load_rgba(buffer, n_rows(), n_cols());

                        m_images.push_back(aimage);
                        ++ cnt;
                }

                return cnt;
        }

        //-------------------------------------------------------------------------------------------------

        bool stl10_task_t::build_folds(const string_t& ifile,
                size_t n_train_images, size_t n_unlabeled_images, size_t n_test_images)
        {
                std::ifstream findices(ifile.c_str());
                if (!findices.is_open())
                {
                        return false;
                }

                for (index_t f = 0; f < n_folds(); f ++)
                {
                        const fold_t train_fold = std::make_pair(f, protocol::train);
                        m_folds[train_fold] = make_image_samples(n_train_images, n_unlabeled_images, 0);

                        string_t line;
                        if (!std::getline(findices, line))
                        {
                                return false;
                        }

                        strings_t tokens;
                        boost::algorithm::split(tokens, line, boost::algorithm::is_any_of(" \t\n\r"));

                        for (index_t t = 0; t < tokens.size(); t ++)
                        {
                                if (tokens[t].empty())
                                {
                                        continue;
                                }

                                try
                                {
                                        const index_t i = text::from_string<index_t>(tokens[t]);
                                        if (i < n_train_images)
                                        {
                                                m_folds[train_fold].push_back(image_sample_t(i, 0));
                                        }
                                        else
                                        {
                                                return false;
                                        }
                                }
                                catch (std::exception&)
                                {
                                        return false;
                                }
                        }
                }

                for (index_t f = 0; f < n_folds(); f ++)
                {
                        const fold_t test_fold = std::make_pair(f, protocol::test);
                        m_folds[test_fold] = make_image_samples(n_train_images + n_unlabeled_images, n_test_images, 0);
                }

                return true;
        }

        //-------------------------------------------------------------------------------------------------

        void stl10_task_t::load(const image_sample_t& isample, sample_t& sample) const
        {
                const annotated_image_t& image = this->image(isample.m_image);
                sample.load_rgba(image, isample);
        }

        //-------------------------------------------------------------------------------------------------
}
