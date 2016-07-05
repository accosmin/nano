#include "softmax.h"

namespace nano
{
        const scalar_t beta = 10;

        softmax_criterion_t::softmax_criterion_t(const string_t& configuration) :
                criterion_t(configuration),
                m_value(0.0)
        {
        }

        void softmax_criterion_t::clear()
        {
                m_value = 0.0;
                m_vgrad.resize(psize());
                m_vgrad.setZero();
        }

        void softmax_criterion_t::accumulate(const scalar_t value)
        {
                m_value += std::exp(value * beta);
        }

        void softmax_criterion_t::accumulate(const vector_t& vgrad, const scalar_t value)
        {
                m_value += std::exp(value * beta);
                m_vgrad += vgrad * std::exp(value * beta);
        }

        void softmax_criterion_t::accumulate(const criterion_t& other)
        {
                assert(dynamic_cast<const softmax_criterion_t*>(&other));
                const softmax_criterion_t& vother = static_cast<const softmax_criterion_t&>(other);
                m_value += vother.m_value;
                m_vgrad += vother.m_vgrad;
        }

        scalar_t softmax_criterion_t::value() const
        {
                assert(count() > 0);

                return std::log(m_value) / beta;
        }

        vector_t softmax_criterion_t::vgrad() const
        {
                assert(count() > 0);

                return m_vgrad / m_value;
        }

        bool softmax_criterion_t::can_regularize() const
        {
                return false;
        }
}

