#include <set>
#include "logger.h"
#include "task_util.h"
#include "math/hash.hpp"
#include "text/to_string.hpp"
#include "vision/image_grid.h"

namespace nano
{
        void describe(const task_t& task)
        {
                log_info()
                << "task [" << task.name()
                << "]: input = " << task.idims() << "x" << task.irows() << "x" << task.icols()
                << ", output = " << task.osize()
                << ", count = " << task.n_samples() << ".";

                for (size_t f = 0; f < task.n_folds(); ++ f)
                {
                        for (auto p : {protocol::train, protocol::valid, protocol::test})
                        {
                                const auto fold = fold_t{f, p};
                                const auto size = task.n_samples(fold);

                                std::map<string_t, size_t> lcounts;
                                for (size_t i = 0; i < size; ++ i)
                                {
                                        lcounts[task.label(fold, i)] ++;
                                }

                                // describe each label separately
                                for (const auto& lcount : lcounts)
                                {
                                        log_info()
                                        << "fold [" << (1 + f) << "," << to_string(p)
                                        << "]: label = " << lcount.first
                                        << ", count = " << lcount.second << "/" << size << "/" << task.n_samples() << ".";
                                }
                        }
                }
        }

        template <typename tvalues>
        static bool has_duplicates(const tvalues& values)
        {
                return std::adjacent_find(values.begin(), values.end()) != values.end();
        }

        template <typename tvalues>
        static bool intersects(const tvalues& values1, const tvalues& values2)
        {
                tvalues intersection;
                std::set_intersection(
                        values1.begin(), values1.end(), values2.begin(), values2.end(),
                        std::back_inserter(intersection));
                return !intersection.empty();
        }

        bool check(const task_t& task)
        {
                using hash_t = std::size_t;
                using hashes_t = std::vector<hash_t>;

                // hash all samples to check folds
                std::map<fold_t, hashes_t> hashes;
                for (size_t f = 0; f < task.n_folds(); ++ f)
                {
                        for (auto p : {protocol::train, protocol::valid, protocol::test})
                        {
                                const auto fold = fold_t{f, p};
                                const auto size = task.n_samples(fold);

                                for (size_t i = 0; i < size; ++ i)
                                {
                                        const auto input = task.input(fold, i);
                                        const auto hash = nano::hash_range(input.data(), input.data() + input.size());
                                        hashes[fold].push_back(hash);
                                }
                        }
                }

                // check if any duplicated samples
                for (auto& h : hashes)
                {
                        std::sort(h.second.begin(), h.second.end());
                        if (has_duplicates(h.second))
                        {
                                return false;
                        }
                }

                // check if the training, validation and tests datasets intersect
                for (size_t f = 0; f < task.n_folds(); ++ f)
                {
                        const auto& train_hashes = hashes[fold_t{f, protocol::train}];
                        const auto& valid_hashes = hashes[fold_t{f, protocol::valid}];
                        const auto& test_hashes = hashes[fold_t{f, protocol::test}];

                        if (    intersects(train_hashes, valid_hashes) ||
                                intersects(valid_hashes, test_hashes) ||
                                intersects(test_hashes, train_hashes))
                        {
                                return false;
                        }
                }

                // OK
                return true;
        }

        void save_as_images(const task_t& task, const fold_t& fold, const string_t& basepath,
                const tensor_size_t tgrows, const tensor_size_t tgcols)
        {
                const auto grows = static_cast<coord_t>(tgrows);
                const auto gcols = static_cast<coord_t>(tgcols);
                const auto border = coord_t{8};
                const auto bkcolor = rgba_t{225, 225, 0, 255};

                const auto size = task.n_samples(fold);

                std::set<string_t> labels;
                for (size_t i = 0; i < size; ++ i)
                {
                        labels.insert(task.label(fold, i));
                }

                // process each label separately
                for (const auto& label : labels)
                {
                        for (size_t i = 0, g = 1; i < size; ++ g)
                        {
                                image_grid_t grid_image(task.irows(), task.icols(), grows, gcols, border, bkcolor);

                                // compose the image block
                                for (coord_t r = 0; r < grows; ++ r)
                                {
                                        for (coord_t c = 0; c < gcols && i < size; ++ c)
                                        {
                                                for (; i < size && label != task.label(fold, i); ++ i) {}

                                                if (i < size)
                                                {
                                                        image_t image;
                                                        image.from_tensor(task.input(fold, i));
                                                        image.make_rgba();
                                                        grid_image.set(r, c, image);
                                                        ++ i;
                                                }
                                        }
                                }

                                // ... and save it
                                const auto path =
                                        basepath +
                                        (label.empty() ? "" : ("_" + label)) + "_group" + to_string(g) + ".png";
                                grid_image.image().save(path);
                        }
                }
        }
}
