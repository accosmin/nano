#include <mutex>
#include "learner.h"
#include "core/logger.h"
#include "core/ibstream.h"
#include "core/obstream.h"
#include "learners/gboost_stump.h"

using namespace nano;

bool learner_t::save(const string_t& path, const string_t& id, const rlearner_t& learner)
{
        obstream_t stream(path);
        return  stream.write(id) &&
                learner->save(stream);
}

rlearner_t learner_t::load(const string_t& path)
{
        ibstream_t stream(path);

        string_t id;
        stream.read(id);

        log_info() << "loading learner id [" << id << "]...";

        auto learner = get_learners().get(id);
        return (learner && learner->load(stream)) ? std::move(learner) : rlearner_t();
}

learner_factory_t& nano::get_learners()
{
        static learner_factory_t manager;

        static std::once_flag flag;
        std::call_once(flag, [] ()
        {
                manager.add<gboost_stump_t>("gboost-stump",     "Gradient Boosting with stumps");
        });

        return manager;
}
