#include "tuner.h"
#include <sstream>
#include <iomanip>
#include "math/random.h"
#include "math/numeric.h"
#include "text/json_writer.h"

using namespace nano;

scalars_t nano::make_pow10_scalars(const scalar_t offset, const int min_power, const int max_power)
{
        assert(min_power <= max_power);

        scalars_t values;
        for (int power = min_power; power <= max_power; ++ power)
        {
                values.push_back(offset + scalar_t(1) * std::pow(scalar_t(10), power));
                values.push_back(offset + scalar_t(3) * std::pow(scalar_t(10), power));
        }

        return values;
}

strings_t tuner_t::get(const size_t max_configs) const
{
        indices_t indices(1u, 0u);
        scalars_t values(m_params.size(), scalar_t(0));

        // generate all possible hyper-parameter combinations
        strings_t configs;
        while (n_params() > 0)
        {
                assert(!indices.empty());

                if (indices.size() == n_params())
                {
                        auto i = indices.size() - 1;
                        for ( ; indices[i] < m_params[i].size(); ++ indices[i])
                        {
                                map(indices, values);
                                configs.push_back(json(values));
                        }

                        for ( ; i > 0; )
                        {
                                -- i;
                                indices.pop_back();
                                if (indices[i] + 1 < m_params[i].size())
                                {
                                        ++ indices[i];
                                        break;
                                }
                                else
                                {
                                        ++ indices[i];
                                }

                        }

                        if (i == 0 && indices[i] == m_params[i].size())
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
        return {configs.begin(), configs.begin() + std::min(configs.size(), max_configs)};
}

void tuner_t::map(const indices_t& indices, scalars_t& values) const
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

string_t tuner_t::json(const scalars_t& values) const
{
        assert(values.size() == m_params.size());

        json_writer_t writer;
        writer.new_object();
        for (size_t i = 0; i < m_params.size(); ++ i)
        {
                const auto& param = m_params[i];
                const auto& value = values[i];

                std::stringstream stream;
                stream << std::fixed << std::setprecision(param.m_precision) << value;

                writer.pair(param.m_name, stream.str());
                if (i + 1 < m_params.size())
                {
                        writer.next();
                }
        }
        writer.end_object();

        return writer.str();
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
