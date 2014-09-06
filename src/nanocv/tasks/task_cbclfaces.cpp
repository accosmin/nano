#include "task_cbclfaces.h"
#include "common/logger.h"
#include "image.h"
#include "loss.h"
#include <fstream>
#include <boost/filesystem.hpp>

namespace ncv
{
        bool cbclfaces_task_t::load(const string_t& dir)
        {
                const string_t train_face_dir = dir + "/train/face/";
                const string_t train_nonface_dir = dir + "/train/non-face/";
                const size_t n_train_samples = 2429 + 4548;

                const string_t test_face_dir = dir + "/test/face/";
                const string_t test_nonface_dir = dir + "/test/non-face/";
                const size_t n_test_samples = 472 + 23573;

                clear_memory(n_train_samples + n_test_samples);

                return  load(train_face_dir, true, protocol::train, 2.0) +
                        load(train_nonface_dir, false, protocol::train, 1.0) == n_train_samples &&
                        load(test_face_dir, true, protocol::test, 50.0) +
                        load(test_nonface_dir, false, protocol::test, 1.0) == n_test_samples;
        }

        size_t cbclfaces_task_t::load(const string_t& dir, bool is_face, protocol p, scalar_t weight)
        {
                log_info() << "CBCL-faces: loading directory <" << dir << "> ...";

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

                                        image_t image;
                                        if (image.load_luma(path.string()))
                                        {
                                                sample_t sample(m_images.size(), sample_region(0, 0), weight);
                                                sample.m_label = is_face ? "face" : "nonface";
                                                sample.m_target = ncv::class_target(is_face ? 0 : 1, n_outputs());
                                                sample.m_fold = { 0, p };
                                                m_samples.push_back(sample);

                                                m_images.push_back(image);

                                                ++ cnt;
                                        }
                                }
                        }
                }

                log_info() << "CBCL-faces: loaded " << cnt << " samples.";

                return cnt;
        }
}
