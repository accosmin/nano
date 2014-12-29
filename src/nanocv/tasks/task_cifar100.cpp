#include "task_cifar100.h"
#include "util/cast.hpp"
#include "file/logger.h"
#include "file/archive.h"
#include "file/stream.h"
#include "loss.h"

namespace ncv
{
        static const string_t tlabels[] =
        {
                "apple",
                "aquarium_fish",
                "baby",
                "bear",
                "beaver",
                "bed",
                "bee",
                "beetle",
                "bicycle",
                "bottle",
                "bowl",
                "boy",
                "bridge",
                "bus",
                "butterfly",
                "camel",
                "can",
                "castle",
                "caterpillar",
                "cattle",
                "chair",
                "chimpanzee",
                "clock",
                "cloud",
                "cockroach",
                "couch",
                "crab",
                "crocodile",
                "cup",
                "dinosaur",
                "dolphin",
                "elephant",
                "flatfish",
                "forest",
                "fox",
                "girl",
                "hamster",
                "house",
                "kangaroo",
                "keyboard",
                "lamp",
                "lawn_mower",
                "leopard",
                "lion",
                "lizard",
                "lobster",
                "man",
                "maple_tree",
                "motorcycle",
                "mountain",
                "mouse",
                "mushroom",
                "oak_tree",
                "orange",
                "orchid",
                "otter",
                "palm_tree",
                "pear",
                "pickup_truck",
                "pine_tree",
                "plain",
                "plate",
                "poppy",
                "porcupine",
                "possum",
                "rabbit",
                "raccoon",
                "ray",
                "road",
                "rocket",
                "rose",
                "sea",
                "seal",
                "shark",
                "shrew",
                "skunk",
                "skyscraper",
                "snail",
                "snake",
                "spider",
                "squirrel",
                "streetcar",
                "sunflower",
                "sweet_pepper",
                "table",
                "tank",
                "telephone",
                "television",
                "tiger",
                "tractor",
                "train",
                "trout",
                "tulip",
                "turtle",
                "wardrobe",
                "whale",
                "willow_tree",
                "wolf",
                "woman",
                "worm"
        };

        cifar100_task_t::cifar100_task_t(const string_t& configuration)
                :       task_t(configuration)
        {
        }

        bool cifar100_task_t::load(const string_t& dir)
        {
                const string_t bfile = dir + "/cifar-100-binary.tar.gz";

                const string_t train_bfile = "train.bin";
                const size_t n_train_samples = 50000;

                const string_t test_bfile = "test.bin";
                const size_t n_test_samples = 10000;

                clear_memory(n_train_samples + n_test_samples);

                const auto op = [&] (const string_t& filename, const io::data_t& data)
                {
                        if (boost::algorithm::iends_with(filename, train_bfile))
                        {
                                return load(filename, data.data(), data.size(), protocol::train, n_train_samples);
                        }
                        else if (boost::algorithm::iends_with(filename, test_bfile))
                        {
                                return load(filename, data.data(), data.size(), protocol::test, n_test_samples);
                        }                        
                        else
                        {
                                return true;
                        }
                };

                log_info() << "CIFAR-100: loading file <" << bfile << "> ...";

                return io::decode(bfile, "CIFAR-100: ", op);
        }
        
        bool cifar100_task_t::load(const string_t& filename, const char* bdata, size_t bdata_size, protocol p, size_t count)
        {
                log_info() << "CIFAR-100: loading file <" << filename << "> ...";                
                
                std::vector<char> buffer(n_rows() * n_cols() * 3);
                char label[2];

                io::stream_t stream(bdata, bdata_size);

                size_t icount = 0;
                while ( stream.read(label, 2) &&       // coarse & fine labels!
                        stream.read(buffer.data(), buffer.size()))
                {
                        const size_t ilabel = math::cast<size_t>(label[1]);
                        if (ilabel >= n_outputs())
                        {
                                continue;
                        }

                        sample_t sample(m_images.size(), sample_region(0, 0));
                        sample.m_label = tlabels[ilabel];
                        sample.m_target = ncv::class_target(ilabel, n_outputs());
                        sample.m_fold = { 0, p };
                        m_samples.push_back(sample);

                        image_t image;
                        image.load_rgba(buffer.data(), n_rows(), n_cols(), n_rows() * n_cols());
                        m_images.push_back(image);

                        icount ++;
                }

                // OK
                log_info() << "CIFAR-100: loaded " << icount << " samples.";
                return (count == icount);
        }
}
