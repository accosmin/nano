#include "task_stl10.h"
#include "common/logger.h"
#include "common/math.hpp"
#include "common/cast.hpp"
#include "loss.h"
#include <fstream>

namespace ncv
{
        static const string_t tlabels[] =
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

        stl10_task_t::stl10_task_t(const string_t& configuration)
                :       task_t(configuration)
        {
        }

        bool stl10_task_t::load(const string_t& dir)
        {
                const string_t train_ifile = dir + "/train_X.bin";
                const string_t train_gfile = dir + "/train_y.bin";
                const string_t train_uifile = dir + "/unlabeled_X.bin";
                const size_t n_train_samples = 10 * 500 + 100000;

                const string_t test_ifile = dir + "/test_X.bin";
                const string_t test_gfile = dir + "/test_y.bin";
                const size_t n_test_samples = 10 * 800;

                const string_t fold_indices_file = dir + "/fold_indices.txt";

                clear_memory(n_train_samples + n_test_samples);

                return  load_binary(train_ifile, train_gfile) +
                        load_binary(train_uifile) == n_train_samples &&
                        load_binary(test_ifile, test_gfile) == n_test_samples &&
                        load_folds(fold_indices_file, 5000, 100000, n_test_samples);
        }
        
        size_t stl10_task_t::load_binary(const string_t& ifile, const string_t& gfile)
        {
                log_info() << "STL-10: loading files <" << ifile << " & " << gfile << "> ...";

                std::ifstream fimage(ifile.c_str(), std::ios::in | std::ios::binary);
                std::ifstream flabel(gfile.c_str(), std::ios::in | std::ios::binary);                
                if (!fimage.is_open() || !flabel.is_open())
                {
                        log_error() << "STL-10: failed to load files!";
                        return false;
                }

                std::vector<char> vbuffer(n_rows() * n_cols() * 3);
                char* buffer = vbuffer.data();
                char label[1];
                
                // load images and annotations
                size_t cnt = 0;
                while ( flabel.read(label, 1) &&
                        fimage.read(buffer, vbuffer.size()))
                {
                        const size_t ilabel = math::cast<size_t>(label[0]) - 1;
                        if (ilabel >= n_outputs())
                        {
                                continue;
                        }

                        sample_t sample(m_images.size(), sample_region(0, 0));
                        sample.m_label = tlabels[ilabel];
                        sample.m_target = ncv::class_target(ilabel, n_outputs());
                        m_samples.push_back(sample);

                        image_t image;
                        image.load_rgba(buffer, n_rows(), n_cols(), n_rows() * n_cols());
                        image.transpose_in_place();
                        m_images.push_back(image);

                        ++ cnt;
                }

                log_info() << "STL-10: loaded " << cnt << " samples.";

                return cnt;
        }

        size_t stl10_task_t::load_binary(const string_t& ifile)
        {
                log_info() << "STL-10: loading file <" << ifile << "> ...";

                std::ifstream fimage(ifile.c_str(), std::ios::in | std::ios::binary);
                if (!fimage.is_open())
                {
                        log_error() << "STL-10: failed to open file!";
                        return false;
                }

                std::vector<char> vbuffer(n_rows() * n_cols() * 3);
                char* buffer = vbuffer.data();

                // load images
                size_t cnt = 0;
                while (fimage.read(buffer, vbuffer.size()))
                {
                        sample_t sample(m_images.size(), sample_region(0, 0));
                        // no annotation
                        m_samples.push_back(sample);

                        image_t image;
                        image.load_rgba(buffer, n_rows(), n_cols(), n_rows() * n_cols());
                        image.transpose_in_place();
                        m_images.push_back(image);

                        ++ cnt;
                }

                log_info() << "STL-10: loaded " << cnt << " samples.";

                return cnt;
        }

        bool stl10_task_t::load_folds(const string_t& ifile, size_t n_train, size_t n_unlabeled, size_t n_test)
        {
                std::ifstream findices(ifile.c_str());
                if (!findices.is_open())
                {
                        return false;
                }

                const samples_t orig_samples = m_samples;
                m_samples.clear();

                const size_t n_folds = 10;

                // training samples [0, n_train) ...
                for (size_t f = 0; f < n_folds; f ++)
                {
                        string_t line;
                        if (!std::getline(findices, line))
                        {
                                return false;
                        }

                        strings_t tokens;
                        text::split(tokens, line, text::is_any_of(" \t\n\r"));

                        for (size_t t = 0; t < tokens.size(); t ++)
                        {
                                if (tokens[t].empty())
                                {
                                        continue;
                                }

                                try
                                {
                                        const size_t i = text::from_string<size_t>(tokens[t]);
                                        if (i < n_train)
                                        {
                                                sample_t sample = orig_samples[i];
                                                sample.m_fold = { f, protocol::train };
                                                m_samples.push_back(sample);
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

                // unlabeled samples [n_train, n_train + n_unlabeled)
                for (size_t f = 0; f < n_folds; f ++)
                {
                        for (size_t i = 0; i < n_unlabeled; i ++)
                        {
                                sample_t sample = orig_samples[n_train + i];
                                sample.m_fold = { f, protocol::train };
                                m_samples.push_back(sample);
                        }
                }

                // testing samples [n_train + n_unlabeled, n_train + n_unlabeled + n_test)
                for (size_t f = 0; f < n_folds; f ++)
                {
                        for (size_t i = 0; i < n_test; i ++)
                        {
                                sample_t sample = orig_samples[n_train + n_unlabeled + i];
                                sample.m_fold = { f, protocol::test };
                                m_samples.push_back(sample);
                        }
                }

                return true;
        }
}
