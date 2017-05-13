#include "iterators/iterator_warp.h"
#include "iterators/iterator_noise.h"

using namespace nano;

iterator_manager_t& nano::get_iterators()
{
        static iterator_manager_t manager;

        static std::once_flag flag;
        std::call_once(flag, [&m = manager] ()
        {
                m.add<iterator_t>("default", "use samples as they are");
                m.add<iterator_warp_t>("warp", "warp image samples (image classification)");
                m.add<iterator_noise_t>("noise", "add random noise to samples (image classification)");
        });

        return manager;
}
