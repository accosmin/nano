#include "task_cbclfaces.h"
#include "common/logger.h"
#include "common/io_archive.h"
#include "image.h"
#include "loss.h"

namespace ncv
{
        cbclfaces_task_t::cbclfaces_task_t(const string_t& configuration)
                :       task_t(configuration)
        {
        }

        bool cbclfaces_task_t::load(const string_t& dir)
        {
                const string_t bfile = dir + "/faces.tar.gz";
                
                const size_t n_train_samples = 2429 + 4548;
                const size_t n_test_samples = 472 + 23573;
        
                clear_memory(n_train_samples + n_test_samples);
                
                const auto op = [&] (const string_t& filename, const io::data_t& data)
                {
                        if (boost::algorithm::iends_with(filename, ".pgm"))
                        {
                                const bool is_face = !boost::algorithm::contains(filename, "non-face");
                                const bool is_test = !boost::algorithm::contains(filename, "train");
                                
                                image_t image;
                                if (image.load_luma(filename, data.data(), data.size()))
                                {
                                        sample_t sample(m_images.size(), sample_region(0, 0));
                                        sample.m_label = is_face ? "face" : "nonface";
                                        sample.m_target = ncv::class_target(is_face ? 0 : 1, n_outputs());
                                        sample.m_fold = { 0, is_test ? protocol::test : protocol::train };
                                        m_samples.push_back(sample);
                                        
                                        m_images.push_back(image);
                                        return true;
                                }
                                else
                                {
                                        return false;
                                }
                        }                        
                        else
                        {                        
                                return true;
                        }
                };
                
                log_info() << "CBCL-faces: loading file <" << bfile << "> ...";

                return  io::decode(bfile, "CBCL-faces: ", op) &&
                        
                        m_samples.size() == n_train_samples + n_test_samples &&
                        m_images.size() == n_train_samples + n_test_samples &&

                        label_normalize(m_samples.begin(), m_samples.begin() + n_train_samples) &&
                        label_normalize(m_samples.begin() + n_train_samples, m_samples.end());
        }
}
