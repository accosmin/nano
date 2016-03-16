#include "task_random.h"
#include "cortex/class.h"
#include "math/clamp.hpp"
#include "math/random.hpp"
#include "tensor/random.hpp"
#include "text/to_string.hpp"
#include "text/from_params.hpp"

namespace nano
{
        random_task_t::random_task_t(const string_t& configuration)
                :       task_t(configuration),
                        m_rows(nano::clamp(nano::from_params<tensor_size_t>(configuration, "rows", 32), 8, 128)),
                        m_cols(nano::clamp(nano::from_params<tensor_size_t>(configuration, "cols", 32), 8, 128)),
                        m_dims(nano::clamp(nano::from_params<tensor_size_t>(configuration, "dims", 2), 2, 10)),
                        m_folds(1),
                        m_color(nano::from_params<color_mode>(configuration, "color", color_mode::rgba)),
                        m_size(nano::clamp(nano::from_params<size_t>(configuration, "size", 1024), 16, 1024 * 1024))
        {
        }

        bool random_task_t::load(const string_t &)
        {
                nano::random_t<size_t> rng_protocol(1, 10);
                nano::random_t<tensor_size_t> rng_output(0, osize() - 1);

                clear_memory(0);

                for (size_t f = 0; f < fsize(); ++ f)
                {
                        for (size_t i = 0; i < m_size; ++ i)
                        {
                                // random protocol: train vs. test (90% training, 10% testing)
                                const protocol p = (rng_protocol() < 9) ? protocol::train : protocol::test;

                                // random output class: character
                                const tensor_index_t o = rng_output();

                                // random image:
                                image_t image(irows(), icols(), color());
                                image.random();

                                // generate image
                                add_image(image);

                                // generate sample
                                sample_t sample(n_images() - 1, sample_region(0, 0));
                                sample.m_label = string_t("class") + nano::to_string(o);
                                sample.m_target = nano::class_target(o, osize());
                                sample.m_fold = {f, p};
                                add_sample(sample);
                        }
                }

                return true;
        }
}
