#include <mutex>
#include "enhancers/enhancer_warp.h"
#include "enhancers/enhancer_noise.h"
#include "enhancers/enhancer_default.h"
#include "enhancers/enhancer_noclass.h"

using namespace nano;

enhancer_factory_t& nano::get_enhancers()
{
        static enhancer_factory_t manager;

        static std::once_flag flag;
        std::call_once(flag, [&m = manager] ()
        {
                m.add<enhancer_warp_t>("warp", "warp image samples (image classification)");
                m.add<enhancer_noise_t>("noise", "add salt&pepper noise to samples");
                m.add<enhancer_default_t>("default", "use samples as they are");
                m.add<enhancer_noclass_t>("noclass", "replace some samples with random samples having no label (classification)");
        });

        return manager;
}
