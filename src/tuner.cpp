#include <cassert>
#include "tuner.h"
#include "math/random.h"
#include "text/json_writer.h"

using namespace nano;

string_t tuner_t::get()
{
        assert(!m_params.empty());
        auto rng = make_rng<scalar_t>(0, 1);

        m_trials.push_back(trial_t{});

        trial_t& trial = *m_trials.rbegin();
        for (const auto& param : m_params)
        {
                const auto value = param.m_min + rng() * (param.m_max - param.m_min);
                trial.m_values.push_back(value);
        }

        return json(trial);
}

void tuner_t::score(const scalar_t score)
{
        assert(!m_trials.empty());
        m_trials.rbegin()->m_score = score;
}

string_t tuner_t::json(const trial_t& trial) const
{
        assert(trial.m_values.size() == m_values.size());

        json_writer_t writer;
        writer.new_object();
        for (size_t i = 0; i < m_params.size(); ++ i)
        {
                const auto& param = m_params[i];
                const auto& value = trial.m_values[i];

                writer.pair(param.m_name, std::pow(param.m_base, value));
                if (i + 1 < m_params.size())
                {
                        writer.next();
                }
        }
        writer.end_object();

        return writer.str();
}
