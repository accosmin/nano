#include <mutex>
#include "model.h"
#include "core/logger.h"
#include "core/ibstream.h"
#include "core/obstream.h"
#include "models/gboost_stump.h"

using namespace nano;

bool model_t::save(const string_t& path, const string_t& id, const model_t& model)
{
        obstream_t stream(path);
        return  stream.write(id) &&
                model.save(stream);
}

rmodel_t model_t::load(const string_t& path)
{
        ibstream_t stream(path);

        string_t id;
        stream.read(id);

        log_info() << "loading model id [" << id << "]...";

        auto model = get_models().get(id);
        return (model && model->load(stream)) ? std::move(model) : rmodel_t();
}

model_factory_t& nano::get_models()
{
        static model_factory_t manager;

        static std::once_flag flag;
        std::call_once(flag, [] ()
        {
                manager.add<gboost_stump_t>("gboost-stump",     "Gradient Boosting with stumps");
        });

        return manager;
}
