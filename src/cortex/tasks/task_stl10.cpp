#include "archive.h"
#include "task_stl10.h"
#include "io/archive.h"
#include "math/cast.hpp"
#include "io/imstream.h"
#include "cortex/class.h"
#include "text/algorithm.h"
#include "cortex/util/logger.h"
#include "text/from_string.hpp"

namespace nano
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

        stl10_task_t::stl10_task_t(const string_t&) :
                mem_vision_task_t("stl-10", 3, 96, 96, 10, 10)
        {
        }

        bool stl10_task_t::populate(const string_t& dir)
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

                const auto op = [&] (const string_t& filename, const nano::buffer_t& data)
                {
                        if (nano::iends_with(filename, train_ifile))
                        {
                                return load_ifile(filename, data.data(), data.size(), false, n_train);
                        }
                        else if (nano::iends_with(filename, train_gfile))
                        {
                                return load_gfile(filename, data.data(), data.size(), n_train);
                        }
                        else if (nano::iends_with(filename, test_ifile))
                        {
                                return load_ifile(filename, data.data(), data.size(), false, n_test);
                        }
                        else if (nano::iends_with(filename, test_gfile))
                        {
                                return load_gfile(filename, data.data(), data.size(), n_test);
                        }
                        else if (nano::iends_with(filename, train_uifile))
                        {
                                return load_ifile(filename, data.data(), data.size(), true, n_unlabeled);
                        }
                        else if (nano::iends_with(filename, fold_file))
                        {
                                return load_folds(filename, data.data(), data.size(), n_test, n_train, n_unlabeled);
                        }
                        else
                        {
                                return true;
                        }
                };
                const auto error_op = [&] (const string_t& message)
                {
                        log_error() << "STL-10: " << message;
                };

                log_info() << "STL-10: loading file <" << bfile << "> ...";

                return nano::unarchive(bfile, op, error_op);
        }

        bool stl10_task_t::load_ifile(const string_t& ifile,
                const char* bdata, const size_t bdata_size, const bool unlabeled, const size_t count)
        {
                log_info() << "STL-10: loading file <" << ifile << "> ...";

                nano::imstream_t stream(bdata, bdata_size);

                const auto buffer_size = irows() * icols() * 3;
                std::vector<char> buffer = nano::make_buffer(buffer_size);

                size_t icount = 0;

                // load images
                while (stream.read(buffer.data(), buffer_size))
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

        bool stl10_task_t::load_gfile(const string_t& gfile,
                const char* bdata, const size_t bdata_size, const size_t count)
        {
                log_info() << "STL-10: loading file <" << gfile << "> ...";

                nano::imstream_t stream(bdata, bdata_size);

                char label;

                size_t iindex = n_images() - count;
                size_t gcount = 0;

                // load annotations
                while (stream.read(&label, 1))
                {
                        const tensor_index_t ilabel = nano::cast<tensor_index_t>(label) - 1;

                        sample_t sample(iindex, sample_region(0, 0));
                        if (ilabel < osize())
                        {
                                sample.m_label = tlabels[ilabel];
                                sample.m_target = nano::class_target(ilabel, osize());
                        }
                        add_sample(sample);

                        ++ gcount;
                        ++ iindex;
                }

                log_info() << "STL-10: loaded " << gcount << " samples.";

                return count == gcount;
        }

        bool stl10_task_t::load_folds(const string_t& ifile, const char* bdata, const size_t bdata_size,
                const size_t n_test, const size_t n_train, const size_t n_unlabeled)
        {
                log_info() << "STL-10: loading file <" << ifile << "> ...";

                // NB: samples arranged line [n_test][n_train][n_unlabeled]

                nano::imstream_t stream(bdata, bdata_size);

                const samples_t orig_samples = this->samples();
                clear_samples(0);

                const size_t fold_size = 1000;

                // training samples [0, n_train) ...
                for (size_t f = 0; f < n_folds; ++ f)
                {
                        string_t line;
                        if (!stream.getline(line))
                        {
                                return false;
                        }

                        const strings_t tokens = nano::split(line, " \t\n\r");

                        size_t fcount = 0;
                        for (size_t t = 0; t < tokens.size(); ++ t)
                        {
                                if (tokens[t].empty())
                                {
                                        continue;
                                }

                                try
                                {
                                        const size_t i = nano::from_string<size_t>(tokens[t]);
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
                for (size_t f = 0; f < n_folds; ++ f)
                {
                        for (size_t i = 0; i < n_unlabeled; ++ i)
                        {
                                sample_t sample = orig_samples[n_test + n_train + i];
                                sample.m_fold = { f, protocol::train };
                                add_sample(sample);
                        }
                }

                // testing samples [n_train + n_unlabeled, n_train + n_unlabeled + n_test)
                for (size_t f = 0; f < n_folds; ++ f)
                {
                        for (size_t i = 0; i < n_test; ++ i)
                        {
                                sample_t sample = orig_samples[i];
                                sample.m_fold = { f, protocol::test };
                                add_sample(sample);
                        }
                }

                return true;
        }
}
