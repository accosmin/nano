#pragma once

#include "../math/clamp.hpp"

namespace ncv
{
        ///
        /// \brief represents a range of scalars
        ///
        template
        <
                typename tscalar_
        >
        class range_t
        {
        public:

                typedef tscalar_        tscalar;

                ///
                /// \brief constructor
                ///
                range_t(tscalar min, tscalar max)
                        :       m_min(min),
                                m_max(max)
                {
                }

                ///
                /// \brief minimum
                ///
                tscalar min() const { return m_min; }

                ///
                /// \brief maximum
                ///
                tscalar max() const { return m_max; }

                ///
                /// \brief clamp the given value to the range
                ///
                tscalar clamp(tscalar value) const { return math::clamp(value, min(), max()); }

        private:

                tscalar         m_min;          ///< minimum
                tscalar         m_max;          ///< maximum
        };
}

