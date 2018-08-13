#include "core/logger.h"
#include "task_cifar100.h"
#include "core/algorithm.h"

using namespace nano;

static const strings_t tlabels =
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
        mem_vision_task_t(make_dims(3, 32, 32), make_dims(100, 1, 1), 10),
        m_dir(string_t(std::getenv("HOME")) + "/experiments/databases/cifar100")
{
}

void cifar100_task_t::from_json(const json_t& json)
{
        nano::from_json(json, "dir", m_dir, "folds", m_folds);
        reconfig(make_dims(3, 32, 32), make_dims(100, 1, 1), m_folds);
}

void cifar100_task_t::to_json(json_t& json) const
{
        nano::to_json(json, "dir", m_dir, "folds", m_folds);
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

        const auto chunk_begin = n_chunks();
        const auto sample_begin = size();

        const auto irows = std::get<1>(idims());
        const auto icols = std::get<2>(idims());
        const auto buffer_size = irows * icols * 3;

        std::vector<char> buffer(static_cast<size_t>(buffer_size));
        char label[2];

        // generate samples
        std::vector<tensor_size_t> ilabels;
        while ( stream.read(label, 2) == 2 &&       // coarse & fine labels!
                stream.read(buffer.data(), buffer_size) == buffer_size &&
                n_chunks() < chunk_begin + count)
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

                ilabels.push_back(ilabel);
        }
        if (ilabels.size() != count)
        {
                log_error() << "CIFAR-100: invalid number of samples!";
                return false;
        }

        // generate folds
        add_samples(p, ilabels, tlabels);

        // OK
        log_info() << "CIFAR-100: loaded " << (n_chunks() - chunk_begin) << " samples.";
        return  n_chunks() == chunk_begin + count &&
                size() == sample_begin + count * m_folds;
}
