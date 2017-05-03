#include <set>
#include "task_util.h"
#include "text/to_string.h"
#include "vision/image_grid.h"
#include "logger.h"

namespace nano
{
        template <typename tvalues>
        static size_t count_duplicates(const tvalues& values)
        {
                size_t count = 0;
                auto it = values.begin();
                while ((it = std::adjacent_find(it, values.end())) != values.end())
                {
                        ++ count;
                        ++ it;
                }
                return count;
        }

        template <typename tvalues>
        static size_t count_intersects(const tvalues& values1, const tvalues& values2)
        {
                tvalues intersection;
                std::set_intersection(
                        values1.begin(), values1.end(), values2.begin(), values2.end(),
                        std::back_inserter(intersection));
                return intersection.size();
        }

        template <typename thashes>
        static void add_hashes(const task_t& task, const fold_t& fold, thashes& hashes)
        {
                const auto size = task.size(fold);
                for (size_t i = 0; i < size; ++ i)
                {
                        hashes.push_back(task.hash(fold, i));
                }
        }

        static size_t count_duplicates(const task_t& task, const size_t f)
        {
                assert(f < task.fsize());

                std::vector<size_t> hashes;

                add_hashes(task, fold_t{f, protocol::train}, hashes);
                add_hashes(task, fold_t{f, protocol::valid}, hashes);
                add_hashes(task, fold_t{f, protocol::test}, hashes);

                return count_duplicates(hashes);
        }

        static size_t count_intersection(const task_t& task, const size_t f)
        {
                assert(f < task.fsize());

                std::vector<size_t> train_hashes;
                std::vector<size_t> valid_hashes;
                std::vector<size_t> test_hashes;

                add_hashes(task, fold_t{f, protocol::train}, train_hashes);
                add_hashes(task, fold_t{f, protocol::valid}, valid_hashes);
                add_hashes(task, fold_t{f, protocol::test}, test_hashes);

                return  std::max(std::max(
                        count_intersects(train_hashes, valid_hashes),
                        count_intersects(valid_hashes, test_hashes)),
                        count_intersects(test_hashes, train_hashes));
        }

        void describe(const task_t& task, const string_t& name)
        {
                log_info() << "task [" << name << "]: in(" << task.idims() << ") -> out(" << task.odims()
                        << "), count = " << task.size() << ".";

                for (size_t f = 0; f < task.fsize(); ++ f)
                {
                        log_info() << "fold [" << (1 + f) << "]: duplicates = " << count_duplicates(task, f) << ".";
                        log_info() << "fold [" << (1 + f) << "]: intersections = " << count_intersection(task, f) << ".";

                        for (const auto p : {protocol::train, protocol::valid, protocol::test})
                        {
                                const auto fold = fold_t{f, p};
                                const auto size = task.size(fold);

                                std::map<string_t, size_t> lcounts;
                                for (size_t i = 0; i < size; ++ i)
                                {
                                        lcounts[task.label(fold, i)] ++;
                                }

                                // describe each label separately
                                for (const auto& lcount : lcounts)
                                {
                                        log_info() << "fold [" << (1 + f) << "," << to_string(p)
                                                << "]: label = " << lcount.first
                                                << ", count = " << lcount.second
                                                << "/" << size << "/" << task.size() << ".";
                                }
                        }
                }
        }

        size_t check_duplicates(const task_t& task)
        {
                size_t max_duplicates = 0;
                for (size_t f = 0; f < task.fsize(); ++ f)
                {
                        max_duplicates = std::max(max_duplicates, count_duplicates(task, f));
                }

                return max_duplicates;
        }

        size_t check_intersection(const task_t& task)
        {
                size_t max_duplicates = 0;
                for (size_t f = 0; f < task.fsize(); ++ f)
                {
                        max_duplicates = std::max(max_duplicates, count_intersection(task, f));
                }

                return max_duplicates;
        }

        void save_as_images(const task_t& task, const fold_t& fold, const string_t& basepath,
                const tensor_size_t tgrows, const tensor_size_t tgcols)
        {
                const auto grows = static_cast<coord_t>(tgrows);
                const auto gcols = static_cast<coord_t>(tgcols);
                const auto border = coord_t{8};
                const auto bkcolor = rgba_t{225, 225, 0, 255};

                const auto size = task.size(fold);
                const auto rows = std::get<1>(task.idims());
                const auto cols = std::get<2>(task.idims());

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
                                image_grid_t grid_image(rows, cols, grows, gcols, border, bkcolor);

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
