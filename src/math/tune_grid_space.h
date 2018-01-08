#pragma once

#include <vector>
#include <cassert>
#include <type_traits>
#include "math/numeric.h"

namespace nano
{
        ///
        /// \brief "space" to search/tune hyper-parameters.
        ///
        template <typename tscalar_>
        struct tune_space_t
        {
                using tscalar = tscalar_;
                using tscalars = std::vector<tscalar>;

                virtual ~tune_space_t() = default;

                ///
                /// \brief return a list of values to try
                ///
                virtual tscalars values() const = 0;

                ///
                /// \brief refine the search "around" the given current optimum (if possible)
                ///
                virtual bool refine(tscalar) = 0;
        };

        template <typename tscalar>
        using tune_spaces_t = std::vector<tune_space_t<tscalar>>;

        ///
        /// \brief finite set of values to search through.
        ///
        template <typename tscalar_>
        struct tune_finite_space_t final : public tune_space_t<tscalar_>
        {
                using tscalar = typename tune_space_t<tscalar_>::tscalar;
                using tscalars = typename tune_space_t<tscalar_>::tscalars;

                explicit tune_finite_space_t(const tscalars& values) :
                        m_values(values)
                {
                }

                tscalars values() const final
                {
                        return m_values;
                }

                bool refine(tscalar) final
                {
                        return false;
                }

        private:

                // attributes
                tscalars        m_values;       ///< list of values to try/evaluate
        };

        template <typename tscalar, typename... tscalars>
        auto make_finite_space(const tscalar param, const tscalars... paramX)
        {
                return tune_finite_space_t<tscalar>({param, paramX...});
        }

        ///
        /// \brief grid space useful for coarse-to-fine searching.
        ///
        template
        <
                typename tscalar_,
                typename tmapping_to_param,
                typename tmapping_from_param,
                typename tvalid_tscalar = typename std::enable_if<std::is_floating_point<tscalar_>::value>::type
        >
        struct tune_grid_space_t final : public tune_space_t<tscalar_>
        {
                using tscalar = typename tune_space_t<tscalar_>::tscalar;
                using tscalars = typename tune_space_t<tscalar_>::tscalars;

                tune_grid_space_t(
                        const tscalar min, const tscalar max, const tscalar epsilon,
                        const tmapping_to_param& to_param,
                        const tmapping_from_param& from_param,
                        const int splits = 6) :
                        m_min(min), m_orig_min(min),
                        m_max(max), m_orig_max(max),
                        m_epsilon(epsilon), m_splits(splits),
                        m_to_param(to_param),
                        m_from_param(from_param)
                {
                        assert(min < max);
                        assert(epsilon > 0);
                        assert(splits > 3);
                }

                tscalars values() const final
                {
                        tscalars values;
                        for (auto i = 0; i <= m_splits; ++ i)
                        {
                                const auto value = m_min + static_cast<tscalar>(i) * delta();
                                values.push_back(m_to_param(value));
                        }
                        return values;
                }

                bool refine(tscalar optimum) final
                {
                        optimum = m_from_param(optimum);

                        const auto var = delta();
                        const auto scale = static_cast<tscalar>(m_splits - 1) / static_cast<tscalar>(m_splits);
                        const auto min = optimum - scale * var;
                        const auto max = optimum + scale * var;

                        m_min = nano::clamp(min, m_orig_min, m_orig_max);
                        m_max = nano::clamp(max, m_orig_min, m_orig_max);

                        return var >= tscalar(1.01) * m_epsilon;
                }

                tscalar delta() const
                {
                        return (m_max - m_min) / static_cast<tscalar>(m_splits);
                }

        private:

                // attributes
                tscalar                 m_min, m_orig_min;
                tscalar                 m_max, m_orig_max;
                tscalar                 m_epsilon;
                int                     m_splits;
                tmapping_to_param       m_to_param;     ///< map between the space values and the parameter values
                tmapping_from_param     m_from_param;   ///< map between the space values and the parameter values
        };

        template <typename tscalar>
        auto make_linear_space(const tscalar min, const tscalar max, const tscalar epsilon, const int splits = 6)
        {
                const auto to_params = [] (const auto x) { return x; };
                const auto from_params = [] (const auto x) { return x; };

                return  tune_grid_space_t<tscalar, decltype(to_params), decltype(from_params)>(
                        min, max, epsilon, to_params, from_params, splits);
        }

        template <typename tscalar>
        auto make_log10_space(const tscalar min, const tscalar max, const tscalar epsilon, const int splits = 6)
        {
                const auto to_params = [] (const auto x) { return std::pow(decltype(x)(10), x); };
                const auto from_params = [] (const auto x) { return std::log10(x); };

                return  tune_grid_space_t<tscalar, decltype(to_params), decltype(from_params)>(
                        min, max, epsilon, to_params, from_params, splits);
        }
}
