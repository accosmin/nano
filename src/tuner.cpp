#include "tuner.h"
#include "math/numeric.h"
#include "text/json_writer.h"

using namespace nano;

namespace
{
        template <typename trng>
        scalar_t urand(const scalar_t min, const scalar_t max, trng&& rng)
        {
                std::uniform_real_distribution<scalar_t> dist(min, max);
                return dist(rng);
        }

        template <typename trng>
        scalar_t urand(const scalar_t min, const scalar_t max, const scalar_t value, const scalar_t scale, trng&& rng)
        {
                const auto delta = scale * (max - min);
                std::normal_distribution<scalar_t> dist(value, delta);
                return clamp(dist(rng), min, max);
        }

        template <typename trng>
        scalar_t urand(const scalars_t& values, trng&& rng)
        {
                std::uniform_int_distribution<size_t> dist(0, values.size() - 1);
                return values[dist(rng)];
        }
}

tuner_t::tuner_t() :
        m_rng(std::random_device{}())
{
}

string_t tuner_t::get()
{
        assert(!m_params.empty());

        const auto trial = explore();
        m_trials.push_back(trial);
        return json(trial);
}

tuner_t::trial_t tuner_t::explore()
{
        trial_t trial;
        trial.m_values.reserve(m_params.size());

        for (const auto& param : m_params)
        {
                switch (param.m_type)
                {
                case param_type::linear:
                case param_type::base10:
                        trial.m_values.push_back(urand(param.m_min, param.m_max, m_rng));
                        break;

                case param_type::finite:
                        assert(!param.m_values.empty());
                        trial.m_values.push_back(urand(param.m_values, m_rng));
                        break;

                default:
                        assert(false);
                }
        }

        return trial;
}

tuner_t::trial_t tuner_t::exploit()
{
        scalars_t scores;
        for (const auto& trial : m_trials)
        {
                scores.push_back(trial.m_score);
        }

        // NB: the score for the world (aka exploration)!
//        scores.push_back(scores.empty() ?
//               scalar_t(1) :
//                *std::max_element(scores.begin(), scores.end()) + std::numeric_limits<scalar_t>::epsilon());
        scores.push_back(
                std::accumulate(scores.begin(), scores.end(), std::numeric_limits<scalar_t>::epsilon()));

        // sample a new trial based on previous trials' scores
        std::discrete_distribution<size_t> dist(scores.begin(), scores.end());
        const auto itrial = dist(m_rng);

        if (itrial < m_trials.size())
        {
                // refine a previous trial
                return exploit(itrial);
        }
        else
        {
                // create a new trial from scratch
                return explore();
        }
}

tuner_t::trial_t tuner_t::exploit(const size_t itrial)
{
        trial_t trial;

        const auto& ptrial = m_trials[itrial];
        const auto scale = std::pow(scalar_t(0.5), static_cast<scalar_t>(ptrial.m_depth));

        for (size_t i = 0; i < m_params.size(); ++ i)
        {
                const auto& param = m_params[i];
                switch (param.m_type)
                {
                case param_type::linear:
                case param_type::base10:
                        trial.m_values.push_back(urand(param.m_min, param.m_max, ptrial.m_values[i], scale, m_rng));
                        break;

                case param_type::finite:
                        assert(!param.m_values.empty());
                        trial.m_values.push_back(ptrial.m_values[i]);
                        break;

                default:
                        assert(false);
                }
        }

        return trial;
}

void tuner_t::score(const scalar_t score)
{
        assert(!m_trials.empty());
        m_trials.rbegin()->m_score = score;
}

string_t tuner_t::optimum() const
{
        assert(!m_trials.empty());
        const auto comp = [] (const auto& t1, const auto& t2) { return t1.m_score < t2.m_score; };
        const auto it = std::max_element(m_trials.begin(), m_trials.end(), comp);
        return json(*it);
}

string_t tuner_t::json(const trial_t& trial) const
{
        assert(trial.m_values.size() == m_params.size());

        json_writer_t writer;
        writer.new_object();
        for (size_t i = 0; i < m_params.size(); ++ i)
        {
                const auto& param = m_params[i];
                const auto& value = trial.m_values[i];

                switch (param.m_type)
                {
                case param_type::linear:        writer.pair(param.m_name, value); break;
                case param_type::finite:        writer.pair(param.m_name, value); break;
                case param_type::base10:        writer.pair(param.m_name, param.m_offset + std::pow(scalar_t(10), value)); break;
                default:                        assert(false);
                }

                if (i + 1 < m_params.size())
                {
                        writer.next();
                }
        }
        writer.end_object();

        return writer.str();
}
