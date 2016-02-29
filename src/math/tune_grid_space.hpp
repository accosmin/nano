#pragma once

#include <vector>
#include <cassert>
#include <type_traits>

namespace math
{
        enum class grid_space
        {
                linear,
                log10
        };

        template
        <
                typename tscalar_,
                typename tsize,
                grid_space space,
                typename tvalid_tscalar = typename std::enable_if<std::is_floating_point<tscalar_>::value>::type
        >
        struct tune_grid_space_t
        {
                using tscalar = tscalar_;
                using tscalars = std::vector<tscalar>;

                tune_grid_space_t(const tscalar min, const tscalar max, const tscalar epsilon, const tsize splits)
                        :       m_min(min), m_max(max), m_epsilon(epsilon), m_splits(splits)
                {
                        assert(min < max);
                        assert(splits > 3);
                        assert(epsilon > 0);
                        assert(epsilon < (max - min) / splits);
                }

                tscalars values() const
                {
                        tscalars values;
                        for (tsize i = 0; i <= m_splits; ++ i)
                        {
                                const auto value = m_min + static_cast<tscalar>(i) * delta();
                                switch (space)
                                {
                                case grid_space::linear:        values.push_back(value); break;
                                case grid_space::log10:         values.push_back(std::pow(tscalar(10), value)); break;
                                default:                        assert(false); break;
                                }
                        }
                        return values;
                }

                bool refine(tscalar optimum)
                {
                        switch (space)
                        {
                        case grid_space::linear:        break;
                        case grid_space::log10:         assert(optimum > 0); optimum = std::log10(optimum); break;
                        default:                        break;
                        }

                        const auto var = delta();
                        m_min = optimum - static_cast<tscalar>(m_splits - 1) * var / static_cast<tscalar>(m_splits);
                        m_max = optimum + static_cast<tscalar>(m_splits - 1) * var / static_cast<tscalar>(m_splits);

                        return var >= tscalar(1.01) * m_epsilon;
                }

                tscalar delta() const
                {
                        return (m_max - m_min) / static_cast<tscalar>(m_splits);
                }

                tscalar         m_min;
                tscalar         m_max;
                tscalar         m_epsilon;
                tsize           m_splits;
        };

        template <typename tscalar, typename tsize>
        auto make_linear_grid_space(const tscalar min, const tscalar max, const tscalar epsilon, const tsize splits)
        {
                return tune_grid_space_t<tscalar, tsize, grid_space::linear>(min, max, epsilon, splits);
        }

        template <typename tscalar, typename tsize>
        auto make_log10_grid_space(const tscalar min, const tscalar max, const tscalar epsilon, const tsize splits)
        {
                return tune_grid_space_t<tscalar, tsize, grid_space::log10>(min, max, epsilon, splits);
        }
}
