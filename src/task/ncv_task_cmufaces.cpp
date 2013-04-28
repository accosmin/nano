#include "ncv_task_cmufaces.h"
#include "ncv_loss.h"
#include "ncv_color.h"
#include "ncv_image.h"
#include <fstream>
#include <boost/filesystem.hpp>

namespace ncv
{
        //-------------------------------------------------------------------------------------------------

        bool cmufaces_task_t::load(const string_t& dir)
        {
                const string_t train_face_dir = dir + "/train/face/";
                const string_t train_nonface_dir = dir + "/train/non-face/";
                const size_t n_train_samples = 2429 + 4548;

                const string_t test_face_dir = dir + "/test/face/";
                const string_t test_nonface_dir = dir + "/test/non-face/";
                const size_t n_test_samples = 472 + 23573;

                m_images.clear();
                m_folds.clear();

                return  load(train_face_dir, true, protocol::train) +
                        load(train_nonface_dir, false, protocol::train) == n_train_samples &&
                        load(test_face_dir, true, protocol::test) +
                        load(test_nonface_dir, false, protocol::test) == n_test_samples &&
                        build_folds(n_train_samples, n_test_samples);
        }

        //-------------------------------------------------------------------------------------------------

        size_t cmufaces_task_t::load(const string_t& dir, bool is_face, protocol p)
        {
                size_t cnt = 0;
                if (    boost::filesystem::exists(dir) &&
                        boost::filesystem::is_directory(dir))
                {
                        const boost::filesystem::directory_iterator it_dir_end;
                        for (boost::filesystem::directory_iterator it_dir(dir); it_dir != it_dir_end; ++ it_dir)
                        {
                                if (boost::filesystem::is_regular_file(it_dir->status()))
                                {
                                        const boost::filesystem::path path(*it_dir);

                                        const annotation_t anno(
                                                static_cast<coord_t>(0),
                                                static_cast<coord_t>(0),
                                                static_cast<coord_t>(n_cols()),
                                                static_cast<coord_t>(n_rows()),
                                                is_face ? "face" : "nonface",
                                                ncv::class_target(is_face ? 0 : 1, n_outputs()));

                                        annotated_image_t aimage;
                                        aimage.m_protocol = p;
                                        aimage.m_annotations.push_back(anno);
                                        if (ncv::load_image(path.string(), aimage.m_image))
                                        {
                                                m_images.push_back(aimage);
                                                ++ cnt;
                                        }
                                }
                        }
                }

                return cnt;
        }

        //-------------------------------------------------------------------------------------------------

        bool cmufaces_task_t::build_folds(size_t n_train_images, size_t n_test_images)
        {
                const fold_t train_fold = std::make_pair(0, protocol::train);
                m_folds[train_fold] = make_image_samples(0, n_train_images, 0);

                const fold_t test_fold = std::make_pair(0, protocol::test);
                m_folds[test_fold] = make_image_samples(n_train_images, n_test_images, 0);

                return true;
        }

        //-------------------------------------------------------------------------------------------------
}
