#include <mutex>
#include "iterators/iterator_warp.h"
#include "iterators/iterator_noise.h"
#include "iterators/iterator_default.h"
#include "iterators/iterator_noclass.h"

using namespace nano;

iterator_factory_t& nano::get_iterators()
{
        static iterator_factory_t manager;

        static std::once_flag flag;
        std::call_once(flag, [&m = manager] ()
        {
                m.add<iterator_warp_t>("warp", "warp image samples (image classification)");
                m.add<iterator_noise_t>("noise", "add salt&pepper noise to samples");
                m.add<iterator_default_t>("default", "use samples as they are");
                m.add<iterator_noclass_t>("noclass", "replace some samples with random samples having no label (classification)");
        });

        return manager;
}
