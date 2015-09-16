#include "task_cifar10.h"
#include "core/class.h"
#include "core/logger.h"
#include "math/cast.hpp"
#include "core/archive.h"
#include "core/mstream.h"
#include "text/ends_with.hpp"

namespace ncv
{
        static const string_t tlabels[] =
        {
                "airplane",
                "automobile",
                "bird",
                "cat",
                "deer",
                "dog",
                "frog",
                "horse",
                "ship",
                "truck"
        };

        cifar10_task_t::cifar10_task_t(const string_t& configuration)
                :       task_t(configuration)
        {
        }

        bool cifar10_task_t::load(const string_t& dir)
        {
                const string_t bfile = dir + "/cifar-10-binary.tar.gz";

                const string_t train_bfile1 = "data_batch_1.bin";
                const string_t train_bfile2 = "data_batch_2.bin";
                const string_t train_bfile3 = "data_batch_3.bin";
                const string_t train_bfile4 = "data_batch_4.bin";
                const string_t train_bfile5 = "data_batch_5.bin";
                const size_t n_train_samples = 10000;// * 5

                const string_t test_bfile = "test_batch.bin";
                const size_t n_test_samples = 10000;

                clear_memory(n_train_samples + n_test_samples);

                const auto op = [&] (const string_t& filename, const buffer_t& data)
                {
                        if (    text::iends_with(filename, train_bfile1) ||
                                text::iends_with(filename, train_bfile2) ||
                                text::iends_with(filename, train_bfile3) ||
                                text::iends_with(filename, train_bfile4) ||
                                text::iends_with(filename, train_bfile5))
                        {
                                return load(filename, data.data(), data.size(), protocol::train, n_train_samples);
                        }
                        else if (text::iends_with(filename, test_bfile))
                        {                                
                                return load(filename, data.data(), data.size(), protocol::test, n_test_samples);
                        }                        
                        else
                        {
                                return true;
                        }
                };

                log_info() << "CIFAR-10: loading file <" << bfile << "> ...";

                return unarchive(bfile, "CIFAR-10: ", op);
        }

        bool cifar10_task_t::load(const string_t& filename, const char* bdata, size_t bdata_size, protocol p, size_t count)
        {
                log_info() << "CIFAR-10: loading file <" << filename << "> ...";
                
                std::vector<char> buffer(irows() * icols() * 3);
                char label[1];

                mstream_t stream(bdata, bdata_size);

                size_t icount = 0;
                while ( stream.read(label, 1) &&
                        stream.read(buffer.data(), buffer.size()))
                {
                        const size_t ilabel = math::cast<size_t>(label[0]);
                        if (ilabel >= osize())
                        {
                                continue;
                        }

                        image_t image;
                        image.load_rgba(buffer.data(), irows(), icols(), irows() * icols());
                        add_image(image);

                        sample_t sample(n_images() - 1, sample_region(0, 0));
                        sample.m_label = tlabels[ilabel];
                        sample.m_target = ncv::class_target(ilabel, osize());
                        sample.m_fold = { 0, p };
                        add_sample(sample);

                        ++ icount;
                }

                // OK
                log_info() << "CIFAR-10: loaded " << icount << " samples.";
                return (count == icount);
        }
}
