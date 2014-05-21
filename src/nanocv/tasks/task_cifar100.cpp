#include "task_cifar100.h"
#include "common/logger.h"
#include "common/math.hpp"
#include "loss.h"
#include <fstream>

namespace ncv
{
        /////////////////////////////////////////////////////////////////////////////////////////

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

        /////////////////////////////////////////////////////////////////////////////////////////

        bool cifar100_task_t::load(const string_t& dir)
        {
                const string_t train_bfile = dir + "/train.bin";
                const size_t n_train_images = 50000;

                const string_t test_bfile = dir + "/test.bin";
                const size_t n_test_images = 10000;

                m_images.clear();
                m_samples.clear();

                return  load(train_bfile, protocol::train) == n_train_images &&
                        load(test_bfile, protocol::test) == n_test_images;
        }

        /////////////////////////////////////////////////////////////////////////////////////////

        size_t cifar100_task_t::load(const string_t& bfile, protocol p)
        {
                log_info() << "CIFAR-100: loading file <" << bfile << "> ...";

                std::ifstream istream(bfile.c_str(), std::ios::in | std::ios::binary);
                if (!istream.is_open())
                {
                        log_error() << "CIFAR-100: failed to open file!";
                        return 0;
                }

                std::vector<char> vbuffer(n_rows() * n_cols() * 3);
                char* buffer = vbuffer.data();
                char label[2];

                // load images and annotations
                size_t cnt = 0;
                while ( istream.read(label, 2) &&       // coarse & fine labels!
                        istream.read(buffer, vbuffer.size()))
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
                        load_rgba(buffer, n_rows(), n_cols(), n_rows() * n_cols(), image);
                        m_images.push_back(image);

                        ++ cnt;
                }

                log_info() << "CIFAR-100: loaded " << cnt << " samples.";

                return cnt;
        }

        /////////////////////////////////////////////////////////////////////////////////////////
}
