#include "class.h"
#include "logger.h"
#include "task_cifar100.h"

namespace nano
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

        cifar100_task_t::cifar100_task_t(const string_t& config) :
                mem_vision_task_t(tensor3d_dims_t{3, 32, 32}, tensor3d_dims_t{100, 1, 1}, 1,
                to_params(config, "dir", string_t(std::getenv("HOME")) + "/experiments/databases/cifar100"))
        {
        }

        bool cifar100_task_t::populate()
        {
                const string_t dir = nano::from_params<string_t>(config(), "dir");

                const string_t bfile = dir + "/cifar-100-binary.tar.gz";

                const string_t train_bfile = "train.bin";
                const size_t n_train_samples = 50000;

                const string_t test_bfile = "test.bin";
                const size_t n_test_samples = 10000;

                const auto op = [&] (const string_t& filename, istream_t& stream)
                {
                        if (nano::ends_with(filename, train_bfile))
                        {
                                return load_binary(filename, stream, protocol::train, n_train_samples);
                        }
                        else if (nano::ends_with(filename, test_bfile))
                        {
                                return load_binary(filename, stream, protocol::test, n_test_samples);
                        }
                        else
                        {
                                return true;
                        }
                };
                const auto error_op = [&] (const string_t& message)
                {
                        log_error() << "CIFAR-100: " << message;
                };

                log_info() << "CIFAR-100: loading file <" << bfile << "> ...";

                return nano::load_archive(bfile, op, error_op);
        }

        bool cifar100_task_t::load_binary(const string_t& filename, istream_t& stream, const protocol p, const size_t count)
        {
                log_info() << "CIFAR-100: loading file <" << filename << "> ...";

                const auto irows = std::get<1>(idims());
                const auto icols = std::get<2>(idims());
                const auto buffer_size = irows * icols * 3;

                std::vector<char> buffer(static_cast<size_t>(buffer_size));
                char label[2];

                size_t icount = 0;
                while ( stream.read(label, 2) == 2 &&       // coarse & fine labels!
                        stream.read(buffer.data(), buffer_size) == buffer_size)
                {
                        const tensor_size_t ilabel = label[1];
                        if (ilabel < 0 || ilabel >= nano::size(odims()))
                        {
                                log_error() << "CIFAR-100: invalid label!";
                                return false;
                        }

                        image_t image;
                        image.load_rgb(buffer.data(), irows, icols, irows * icols);
                        add_chunk(image, image.hash());

                        const auto fold = make_fold(0, p);
                        add_sample(fold, n_chunks() - 1, class_target(ilabel, nano::size(odims())), tlabels[ilabel]);

                        icount ++;
                }

                // OK
                log_info() << "CIFAR-100: loaded " << icount << " samples.";
                return (count == icount);
        }
}
