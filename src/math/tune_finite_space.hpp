#pragma once

#include <vector>

namespace math
{
        ///
        /// \brief finite set of values to search through.
        ///
        template
        <
                typename tscalar_
        >
        class tune_finite_space_t
        {
        public:
                using tscalar = tscalar_;
                using tscalars = std::vector<tscalar>;

                explicit tune_finite_space_t(const tscalars& values) : m_values(values)
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

        private:

                tscalars        m_values;
        };

        template
        <
                typename tscalar,
                typename... tscalars
        >
        auto make_finite_space(const tscalar param, const tscalars... paramX)
        {
                return tune_finite_space_t<tscalar>({param, paramX...});
        }
}
