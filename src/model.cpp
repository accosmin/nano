#include <mutex>
#include "model.h"
#include "core/table.h"
#include "core/tpool.h"
#include "core/logger.h"
#include "core/ibstream.h"
#include "core/obstream.h"
#include "models/gboost_stump.h"

using namespace nano;

training_t::status training_t::update(const training_t::state_t& state, const int patience)
{
        // out-of-bounds values (e.g. caused by invalid line-search steps)
        if (!state)
        {
                return (m_status = status::diverge);
        }

        m_history.push_back(state);

        // improved performance
        if (state < m_optimum)
        {
                m_optimum = state;
                return (m_status = status::better);
        }

        // worse performance, but not enough epochs, keep training
        else if (state.m_epoch < patience)
        {
                return (m_status = status::worse);
        }

        // last improvement not far in the past, keep training
        else if (state.m_epoch < m_optimum.m_epoch + patience)
        {
                return (m_status = status::worse);
        }

        // no improvement since many epochs, overfitting detected
        else
        {
                return (m_status = status::overfit);
        }
}

scalar_t training_t::convergence_speed() const
{
        const auto op = [](const auto& prv, const auto& crt)
        {
                assert(crt.m_train.m_value >= scalar_t(0));
                assert(prv.m_train.m_value >= scalar_t(0));
                assert(crt.m_milis >= prv.m_milis);

                const auto ratio_eps = nano::epsilon0<scalar_t>();
                const auto ratio = (ratio_eps + crt.m_train.m_value) / (ratio_eps + prv.m_train.m_value);
                const auto delta = 1 + crt.m_milis.count() - prv.m_milis.count();

                // convergence speed ~ loss decrease ratio / second
                const auto ret = static_cast<scalar_t>(std::pow(
                        static_cast<double>(ratio),
                        static_cast<double>(1000) / static_cast<double>(delta)));
                return std::isfinite(ret) ? nano::clamp(ret, scalar_t(0), scalar_t(1)) : scalar_t(1);
        };

        nano::stats_t speeds;
        for (size_t i = 0; i + 1 < m_history.size(); ++ i)
        {
                speeds(op(m_history[i], m_history[i + 1]));
        }

        return static_cast<scalar_t>(speeds.avg());
}

bool training_t::save(const string_t& path) const
{
        table_t table;

        auto&& header = table.header();
        header  << "epoch"
                << "train_loss" << "train_error"
                << "valid_loss" << "valid_error"
                << "test_loss" << "test_error"
                << "seconds";

        int index = 0;
        for (const auto& state : history())
        {
                auto&& row = table.append();
                row     << (index ++)
                        << state.m_train.m_value << state.m_train.m_error
                        << state.m_valid.m_value << state.m_valid.m_error
                        << state.m_test.m_value << state.m_test.m_error
                        << idiv(state.m_milis.count(), 1000);
        }

        return table.save(path);
}

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
