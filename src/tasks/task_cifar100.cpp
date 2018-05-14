#include "logger.h"
#include "task_cifar100.h"
#include "text/algorithm.h"

using namespace nano;

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

cifar100_task_t::cifar100_task_t() :
        mem_vision_task_t(tensor3d_dim_t{3, 32, 32}, tensor3d_dim_t{100, 1, 1}, 1),
        m_dir(string_t(std::getenv("HOME")) + "/experiments/databases/cifar100")
{
}

void cifar100_task_t::from_json(const json_t& json)
{
        nano::from_json(json, "dir", m_dir);
}

void cifar100_task_t::to_json(json_t& json) const
{
        nano::to_json(json, "dir", m_dir);
}

bool cifar100_task_t::populate()
{
        const auto bfile = m_dir + "/cifar-100-binary.tar.gz";

        const auto train_bfile = "train.bin";
        const size_t n_train_samples = 50000;

        const auto test_bfile = "test.bin";
        const size_t n_test_samples = 10000;

        const auto op = [&] (const string_t& filename, istream_t& stream)
        {
                if (nano::ends_with(filename, train_bfile))
                {
                        return load_binary(filename, stream, split2(n_train_samples, protocol::train, 80, protocol::valid));
                }
                else if (nano::ends_with(filename, test_bfile))
                {
                        return load_binary(filename, stream, std::vector<protocol>(n_test_samples, protocol::test));
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

bool cifar100_task_t::load_binary(const string_t& filename, istream_t& stream, const std::vector<protocol>& protocols)
{
        log_info() << "CIFAR-100: loading file <" << filename << "> ...";

        const auto irows = std::get<1>(idims());
        const auto icols = std::get<2>(idims());
        const auto buffer_size = irows * icols * 3;

        std::vector<char> buffer(static_cast<size_t>(buffer_size));
        char label[2];

        size_t icount = 0;
        while ( stream.read(label, 2) == 2 &&       // coarse & fine labels!
                stream.read(buffer.data(), buffer_size) == buffer_size &&
                icount < protocols.size())
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

                const auto fold = fold_t{0, protocols[icount ++]};
                add_sample(fold, n_chunks() - 1, class_target(ilabel, nano::size(odims())), tlabels[ilabel]);
        }

        // OK
        log_info() << "CIFAR-100: loaded " << icount << " samples.";
        return (protocols.size() == icount);
}
