#include <mutex>
#include "models/forward_network.h"

using namespace nano;

model_factory_t& nano::get_models()
{
        static model_factory_t manager;

        static std::once_flag flag;
        std::call_once(flag, [&m = manager] ()
        {
                m.add<forward_network_t>("forward-network", "feed-forward network");
        });

        return manager;
}
