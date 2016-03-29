#include <set>
#include "task.h"
#include "logger.h"
#include "text/to_string.hpp"
#include "vision/image_grid.h"

namespace nano
{
        task_manager_t& get_tasks()
        {
                static task_manager_t manager;
                return manager;
        }

        void task_t::describe() const
        {
                log_info()
                        << "task [" << name() << "]: input = " << idims() << "x" << irows() << "x" << icols()
                        << ", output = " << osize()
                        << ", count = " << n_samples() << ".";

                for (size_t f = 0; f < n_folds(); ++ f)
                {
                        for (auto p : {protocol::train, protocol::valid, protocol::test})
                        {
                                const auto fold = fold_t{f, p};
                                const auto size = n_samples(fold);

                                std::map<string_t, size_t> lcounts;
                                for (size_t i = 0; i < size; ++ i)
                                {
                                        lcounts[label(fold, i)] ++;
                                }

                                // describe each label separately
                                for (const auto& lcount : lcounts)
                                {
                                        log_info()
                                                << "fold [" << (1 + f) << "," << to_string(p)
                                                << "]: label = " << lcount.first
                                                << ", count = " << lcount.second << "/" << size << "/" << n_samples() << ".";
                                }
                        }
                }
        }

        void task_t::save_as_images(const fold_t& fold, const string_t& basepath,
                const tensor_size_t tgrows, const tensor_size_t tgcols) const
        {
                const auto grows = static_cast<coord_t>(tgrows);
                const auto gcols = static_cast<coord_t>(tgcols);
                const auto border = coord_t{8};
                const auto bkcolor = rgba_t{225, 225, 0, 255};

                const auto size = n_samples(fold);

                std::set<string_t> labels;
                for (size_t i = 0; i < size; ++ i)
                {
                        labels.insert(label(fold, i));
                }

                // process each label separately
                for (const auto& label : labels)
                {
                        for (size_t i = 0, g = 1; i < size; ++ g)
                        {
                                image_grid_t grid_image(irows(), icols(), grows, gcols, border, bkcolor);

                                // compose the image block
                                for (coord_t r = 0; r < grows; ++ r)
                                {
                                        for (coord_t c = 0; c < gcols && i < size; ++ c)
                                        {
                                                for (; i < size && label != this->label(fold, i); ++ i) {}

                                                if (i < size)
                                                {
                                                        image_t image;
                                                        image.from_tensor(input(fold, i));
                                                        image.make_rgba();
                                                        grid_image.set(r, c, image);
                                                        ++ i;
                                                }
                                        }
                                }

                                // ... and save it
                                const auto path = basepath + (label.empty() ? "" : ("_" + label)) + "_group" + to_string(g) + ".png";
                                grid_image.image().save(path);
                        }
                }
        }
}
