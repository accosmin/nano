#include <set>
#include "logger.h"
#include "task_util.h"
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
