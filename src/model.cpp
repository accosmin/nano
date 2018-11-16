#include <mutex>
#include "model.h"
#include "core/tpool.h"
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

model_t::evaluate_t model_t::evaluate(const task_t& task, const fold_t& fold, const loss_t& loss) const
{
        const auto& tpool = tpool_t::instance();

        std::vector<stats_t> errors(tpool.workers());
        std::vector<stats_t> values(tpool.workers());

        const timer_t timer;
        loopit(task.size(fold), [&] (const size_t i, const size_t t)
        {
                const auto input = task.input(fold, i);
                const auto target = task.target(fold, i);
                const auto output = this->output(input);

                assert(t < tpool.workers());
                errors[t](loss.error(target, output));
                values[t](loss.value(target, output));
        });

        const auto millis = timer.milliseconds().count();

        evaluate_t ret;
        for (size_t t = 0; t < tpool.workers(); ++ t)
        {
                ret.m_errors(errors[t]);
                ret.m_values(values[t]);
        }
        ret.m_millis = milliseconds_t{idiv(millis, task.size(fold))};

        return ret;
}
