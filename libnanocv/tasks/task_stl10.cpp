#include "task_stl10.h"
#include "libcore/class.h"
#include "libcore/logger.h"
#include "libmath/cast.hpp"
#include "libtext/split.hpp"
#include "libtext/ends_with.hpp"
#include "libtext/from_string.hpp"
#include "libnanocv/file/stream.h"
#include "libnanocv/file/archive.h"

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
                const string_t bfile = dir + "/stl10_binary.tar.gz";
                
                const string_t train_ifile = "train_X.bin";
                const string_t train_gfile = "train_y.bin";
                const string_t train_uifile = "unlabeled_X.bin";
                const size_t n_train = 5000;
                const size_t n_unlabeled = 100000;

                const string_t test_ifile = "test_X.bin";
                const string_t test_gfile = "test_y.bin";
                const size_t n_test = 8000;

                const string_t fold_file = "fold_indices.txt";

                clear_memory(n_test + n_train + n_unlabeled);
                
                const auto op = [&] (const string_t& filename, const io::buffer_t& data)
                {
                        if (text::iends_with(filename, train_ifile))
                        {
                                return load_ifile(filename, data.data(), data.size(), false, n_train);
                        }
                        else if (text::iends_with(filename, train_gfile))
                        {
                                return load_gfile(filename, data.data(), data.size(), n_train);
                        }
                        else if (text::iends_with(filename, test_ifile))
                        {
                                return load_ifile(filename, data.data(), data.size(), false, n_test);
                        }
                        else if (text::iends_with(filename, test_gfile))
                        {
                                return load_gfile(filename, data.data(), data.size(), n_test);
                        }
                        else if (text::iends_with(filename, train_uifile))
                        {
                                return load_ifile(filename, data.data(), data.size(), true, n_unlabeled);
                        }
                        else if (text::iends_with(filename, fold_file))
                        {
                                return load_folds(filename, data.data(), data.size(), n_test, n_train, n_unlabeled);
                        }
                        else
                        {                        
                                return true;
                        }
                };
                
                log_info() << "STL-10: loading file <" << bfile << "> ...";

                return  io::decode(bfile, "STL-10: ", op);
        }
        
        bool stl10_task_t::load_ifile(const string_t& ifile, const char* bdata, size_t bdata_size, bool unlabeled, size_t count)
        {
                log_info() << "STL-10: loading file <" << ifile << "> ...";

                io::stream_t stream(bdata, bdata_size);

                std::vector<char> buffer(irows() * icols() * 3);

                size_t icount = 0;
                
                // load images
                while (stream.read(buffer.data(), buffer.size()))
                {
                        image_t image;
                        image.load_rgba(buffer.data(), irows(), icols(), irows() * icols());
                        image.transpose_in_place();
                        add_image(image);

                        if (unlabeled)
                        {
                                sample_t sample(n_images() - 1, sample_region(0, 0));
                                // no annotation
                                add_sample(sample);
                        }

                        ++ icount;
                }

                log_info() << "STL-10: loaded " << icount << " images.";

                return count == icount;
        }

        bool stl10_task_t::load_gfile(const string_t& gfile, const char* bdata, size_t bdata_size, size_t count)
        {
                log_info() << "STL-10: loading file <" << gfile << "> ...";

                io::stream_t stream(bdata, bdata_size);

                char label;

                size_t iindex = n_images() - count;
                size_t gcount = 0;

                // load annotations
                while (stream.read(&label, 1))
                {
                        const size_t ilabel = math::cast<size_t>(label) - 1;

                        sample_t sample(iindex, sample_region(0, 0));
                        if (ilabel < osize())
                        {
                                sample.m_label = tlabels[ilabel];
                                sample.m_target = ncv::class_target(ilabel, osize());
                        }
                        add_sample(sample);

                        ++ gcount;
                        ++ iindex;
                }

                log_info() << "STL-10: loaded " << gcount << " samples.";

                return count == gcount;
        }
        
        bool stl10_task_t::load_folds(const string_t& ifile, const char* bdata, size_t bdata_size, 
                size_t n_test, size_t n_train, size_t n_unlabeled)
        {
                log_info() << "STL-10: loading file <" << ifile << "> ...";

                // NB: samples arranged line [n_test][n_train][n_unlabeled]

                io::stream_t stream(bdata, bdata_size);
                
                const samples_t orig_samples = this->samples();
                clear_samples(0);

                const size_t n_folds = 10;
                const size_t fold_size = 1000;

                // training samples [0, n_train) ...
                for (size_t f = 0; f < n_folds; f ++)
                {
                        string_t line;
                        if (!stream.getline(line))
                        {
                                return false;
                        }

                        const strings_t tokens = text::split(line, " \t\n\r");

                        size_t fcount = 0;
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
                                                sample_t sample = orig_samples[n_test + i];
                                                sample.m_fold = { f, protocol::train };
                                                add_sample(sample);

                                                ++ fcount;
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

                        log_info() << "STL-10: loaded " << fcount << " samples for fold [" << (f + 1) << "/" << n_folds << "].";
                        if (fcount != fold_size)
                        {
                                return false;
                        }
                }

                // unlabeled samples [n_train, n_train + n_unlabeled)
                for (size_t f = 0; f < n_folds; f ++)
                {
                        for (size_t i = 0; i < n_unlabeled; i ++)
                        {
                                sample_t sample = orig_samples[n_test + n_train + i];
                                sample.m_fold = { f, protocol::train };
                                add_sample(sample);
                        }
                }

                // testing samples [n_train + n_unlabeled, n_train + n_unlabeled + n_test)
                for (size_t f = 0; f < n_folds; f ++)
                {
                        for (size_t i = 0; i < n_test; i ++)
                        {
                                sample_t sample = orig_samples[i];
                                sample.m_fold = { f, protocol::test };
                                add_sample(sample);
                        }
                }

                return true;
        }
}
