#pragma once

#include <vector>
#include <cassert>
#include <type_traits>
#include "math/clamp.hpp"

namespace zob
{
        ///
        /// \brief map values from the search/tuning space to hyper-parameters.
        ///
        template
        <
                typename tto_param,
                typename tfrom_param
        >
        struct mapping_t
        {
                mapping_t(const tto_param& to_param, const tfrom_param& from_param)
                        :       m_to_param(to_param), m_from_param(from_param)
                {
                }

                template <typename tscalar>
                tscalar to_param(const tscalar x) const
                {
                        return m_to_param(x);
                }

                template <typename tscalar>
                tscalar from_param(const tscalar x) const
                {
                        return m_from_param(x);
                }

                tto_param       m_to_param;
                tfrom_param     m_from_param;
        };

        template <typename tto_param, typename tfrom_param>
        auto make_mapping(const tto_param& to_param, const tfrom_param& from_param)
        {
                return mapping_t<tto_param, tfrom_param>(to_param, from_param);
        }

        inline auto make_identity_mapping()
        {
                return  make_mapping(
                        [] (const auto x) { return x; },
                        [] (const auto x) { return x; });
        }

        inline auto make_log10_mapping()
        {
                return  make_mapping(
                        [] (const auto x) { return std::pow(decltype(x)(10), x); },
                        [] (const auto x) { return std::log10(x); });
        }

        ///
        /// \brief grid space useful for coarse-to-fine searching.
        ///
        template
        <
                typename tscalar_,
                typename tmapping,
                typename tvalid_tscalar = typename std::enable_if<std::is_floating_point<tscalar_>::value>::type
        >
        class tune_grid_space_t
        {
        public:
                using tscalar = tscalar_;
                using tscalars = std::vector<tscalar>;

                tune_grid_space_t(
                        const tscalar min, const tscalar max, const tscalar epsilon, const tmapping& mapping,
                        const int splits = 4)
                        :       m_min(min), m_orig_min(min),
                                m_max(max), m_orig_max(max),
                                m_epsilon(epsilon), m_splits(splits), m_mapping(mapping)
                {
                        assert(min < max);
                        assert(epsilon > 0);
                        assert(splits > 3);
                        assert(epsilon < (max - min) / m_splits);
                }

                tscalars values() const
                {
                        tscalars values;
                        for (auto i = 0; i <= m_splits; ++ i)
                        {
                                const auto value = m_min + i * delta();
                                values.push_back(m_mapping.to_param(value));
                        }
                        return values;
                }

                bool refine(tscalar optimum)
                {
                        optimum = m_mapping.from_param(optimum);

                        const auto var = delta();
                        const auto min = optimum - (m_splits - 1) * var / m_splits;
                        const auto max = optimum + (m_splits - 1) * var / m_splits;

                        m_min = zob::clamp(min, m_orig_min, m_orig_max);
                        m_max = zob::clamp(max, m_orig_min, m_orig_max);

                        return var >= tscalar(1.01) * m_epsilon;
                }

                tscalar delta() const
                {
                        return (m_max - m_min) / static_cast<tscalar>(m_splits);
                }

        private:

                // attributes
                tscalar         m_min, m_orig_min;
                tscalar         m_max, m_orig_max;
                tscalar         m_epsilon;
                int             m_splits;
                tmapping        m_mapping;      ///< map between the space values and the parameter values
        };

        template <typename tscalar>
        auto make_linear_space(const tscalar min, const tscalar max, const tscalar epsilon, const int splits = 4)
        {
                const auto mapping = make_identity_mapping();
                return tune_grid_space_t<tscalar, decltype(mapping)>(min, max, epsilon, mapping, splits);
        }

        template <typename tscalar>
        auto make_log10_space(const tscalar min, const tscalar max, const tscalar epsilon, const int splits = 4)
        {
                const auto mapping = make_log10_mapping();
                return tune_grid_space_t<tscalar, decltype(mapping)>(min, max, epsilon, mapping, splits);
        }
}
