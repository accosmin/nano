#pragma once

#include <vector>

namespace math
{
        template
        <
                typename tscalar_
        >
        struct tune_finite_space_t
        {
                using tscalar = tscalar_;
                using tscalars = std::vector<tscalar>;

                template
                <
                        typename tscalars_
                >
                tune_finite_space_t(const tscalars_& values) : m_values(values)
                {
                }

                auto values() const
                {
                        return m_values;
                }

                bool refine(const tscalar)
                {
                        return false;
                }

                tscalars        m_values;
        };

        template
        <
                typename tscalar
        >
        auto make_finite_space(const std::initializer_list<tscalar>& scalars)
        {
                return tune_finite_space_t<tscalar>(scalars);
        }
}
