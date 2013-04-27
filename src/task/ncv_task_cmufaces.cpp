#include "ncv_task_cmufaces.h"
#include "ncv_loss.h"
#include "ncv_color.h"
#include "ncv_image.h"
#include <fstream>
#include <boost/filesystem.hpp>

namespace ncv
{
        //-------------------------------------------------------------------------------------------------

        bool cmufaces_task::load(const string_t& dir)
        {
                const string_t train_face_dir = dir + "/train/face/";
                const string_t train_nonface_dir = dir + "/train/non-face/";
                const size_t train_n_samples = 2429 + 4548;

                const string_t test_face_dir = dir + "/test/face/";
                const string_t test_nonface_dir = dir + "/test/non-face/";
                const size_t test_n_samples = 472 + 23573;

                m_images.clear();

                return  load(train_face_dir, true, protocol::train) +
                        load(train_nonface_dir, false, protocol::train) == train_n_samples &&

                        load(test_face_dir, true, protocol::test) +
                        load(test_nonface_dir, false, protocol::test) == test_n_samples;
        }

        //-------------------------------------------------------------------------------------------------

        size_t cmufaces_task::fold_size(index_t /*f*/, protocol p) const
        {
                switch (p)
                {
                case protocol::train:
                        return 0;
                }
        }

        //-------------------------------------------------------------------------------------------------

        bool cmufaces_task::fold_sample(index_t /*f*/, protocol p, index_t s, sample& ss) const
        {

        }

        //-------------------------------------------------------------------------------------------------

        size_t cmufaces_task::load(const string_t& dir, bool is_face, protocol p)
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

                                        annotation anno(static_cast<coord_t>(0),
                                                        static_cast<coord_t>(0),
                                                        static_cast<coord_t>(n_cols()),
                                                        static_cast<coord_t>(n_rows()),
                                                        is_face ? "face" : "nonface",
                                                        ncv::class_target(is_face ? 0 : 1, n_outputs()));

                                        annotated_image aimage;
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
}
