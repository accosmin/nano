#pragma once

#include <cassert>

namespace math
{
        enum class average
        {
                vanilla,                ///< empirical average
                weighted,               ///< empirical weighted-average
                geometric,              ///< geometrical average (using a fixed momentum)
        };

        template <typename tscalar>
        struct average_vanilla_accumulator_t
        {
                average_vanilla_accumulator_t() : m_denominator(0)
                {
                }

                void operator()(tscalar& average, const tscalar value)
                {
                        average = (average * m_denominator + value) / (m_denominator + tscalar(1));
                        m_denominator += tscalar(1);
                }

                tscalar m_denominator;
        };

        template <typename tscalar>
        struct average_weighted_accumulator_t
        {
                average_weighted_accumulator_t() : m_denominator(0)
                {
                }

                void operator()(tscalar& average, const tscalar value, const tscalar weight)
                {
                        average = (average * m_denominator + value * weight) / (m_denominator + weight);
                        m_denominator += weight;
                }

                tscalar m_denominator;
        };

        template <typename tscalar>
        struct average_geometric_accumulator_t
        {
                explicit average_geometric_accumulator_t(const tscalar momentum) : m_momentum(momentum)
                {
                        assert(m_momentum > tscalar(0));
                        assert(m_momentum < tscalar(1));
                }

                void operator()(tscalar& average, const tscalar value) const
                {
                        average = m_momentum * average + (tscalar(1) - m_momentum) * value;
                }

                tscalar m_momentum;
        };
}

