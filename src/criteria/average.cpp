#include "average.h"

namespace nano
{
        average_criterion_t::average_criterion_t(const string_t& configuration) :
                criterion_t(configuration),
                m_value(0.0)
        {
        }

        rcriterion_t average_criterion_t::clone(const string_t& configuration) const
        {
                return std::make_unique<average_criterion_t>(configuration);
        }

        rcriterion_t average_criterion_t::clone() const
        {
                return std::make_unique<average_criterion_t>(*this);
        }

        void average_criterion_t::clear()
        {
                m_value = 0.0;
                m_vgrad.resize(psize());
                m_vgrad.setZero();
        }

        void average_criterion_t::accumulate(scalar_t value)
        {
                m_value += value;
        }

        void average_criterion_t::accumulate(const vector_t& vgrad, scalar_t value)
        {
                m_value += value;
                m_vgrad += vgrad;
        }

        void average_criterion_t::accumulate(const criterion_t& other)
        {
                assert(dynamic_cast<const average_criterion_t*>(&other));
                const average_criterion_t& vother = static_cast<const average_criterion_t&>(other);
                m_value += vother.m_value;
                m_vgrad += vother.m_vgrad;
        }

        scalar_t average_criterion_t::value() const
        {
                assert(count() > 0);

                return m_value / static_cast<scalar_t>(count());
        }

        vector_t average_criterion_t::vgrad() const
        {
                assert(count() > 0);

                return m_vgrad / static_cast<scalar_t>(count());
        }

        bool average_criterion_t::can_regularize() const
        {
                return false;
        }
}

