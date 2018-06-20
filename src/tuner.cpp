#include "tuner.h"
#include <sstream>
#include <iomanip>
#include "text/json.h"
#include "math/random.h"
#include "math/numeric.h"

using namespace nano;

void tuner_t::add_pow10s(const char* name, const scalar_t offset, const int min_power, const int max_power)
{
        assert(min_power <= max_power);

        const auto precision = std::max(std::abs(min_power), std::abs(max_power));

        strings_t values;
        for (int power = min_power; power <= max_power; ++ power)
        {
                for (const auto scale : {1, 3})
                {
                        std::stringstream stream;
                        stream << std::fixed << std::setprecision(precision)
                                << (offset + scalar_t(scale) * std::pow(scalar_t(10), power));
                        values.push_back(stream.str());
                }
        }

        m_params.emplace_back(name, std::move(values));
}

jsons_t tuner_t::get(const size_t max_configs) const
{
        indices_t indices(1u, 0u);
        strings_t values(m_params.size());

        // generate all possible hyper-parameter combinations
        jsons_t configs;
        while (n_params() > 0)
        {
                assert(!indices.empty());

                if (indices.size() == n_params())
                {
                        auto i = indices.size() - 1;
                        for ( ; indices[i] < m_params[i].m_values.size(); ++ indices[i])
                        {
                                map(indices, values);
                                configs.push_back(json(values));
                        }

                        for ( ; i > 0; )
                        {
                                -- i;
                                indices.pop_back();
                                if (indices[i] + 1 < m_params[i].m_values.size())
                                {
                                        ++ indices[i];
                                        break;
                                }
                                else
                                {
                                        ++ indices[i];
                                }

                        }

                        if (i == 0 && indices[i] == m_params[i].m_values.size())
                                break;
                }
                else
                {
                        indices.push_back(0u);
                }
        }

        assert(configs.size() == n_configs());

        // randomly return the required number of hyper-parameter combinations
        std::shuffle(configs.begin(), configs.end(), make_rng());
        configs.erase(configs.begin() + std::min(configs.size(), max_configs), configs.end());
        return configs;
}

void tuner_t::map(const indices_t& indices, strings_t& values) const
{
        assert(indices.size() == m_params.size());
        assert(values.size() == m_params.size());

        for (size_t i = 0; i < m_params.size(); ++ i)
        {
                const auto& param = m_params[i];

                assert(indices[i] < param.m_values.size());
                values[i] = param.m_values[indices[i]];
        }
}

json_t tuner_t::json(const strings_t& values) const
{
        assert(values.size() == m_params.size());

        json_t json;
        for (size_t i = 0; i < m_params.size(); ++ i)
        {
                const auto& param = m_params[i];
                const auto& value = values[i];

                json[param.m_name] = value;
        }

        return json;
}

size_t tuner_t::n_configs() const
{
        size_t count = 1;
        for (const auto& param : m_params)
        {
                count *= param.m_values.size();
        }

        return count;
}
