#include "softmax.h"
#include "math/numeric.h"
#include "text/to_params.h"
#include "text/from_params.h"

namespace nano
{
        softmax_criterion_t::softmax_criterion_t(const string_t& configuration) :
                criterion_t(to_params(configuration, "beta", "5[1,10]")),
                m_beta(clamp(from_params(config(), "beta", scalar_t(5)), scalar_t(1), scalar_t(10))),
                m_value(0)
        {
        }

        rcriterion_t softmax_criterion_t::clone() const
        {
                return std::make_unique<softmax_criterion_t>(*this);
        }

        void softmax_criterion_t::clear()
        {
                criterion_t::clear();

                m_value = 0;
                m_vgrad.resize(psize());
                m_vgrad.setZero();
        }

        void softmax_criterion_t::accumulate(const scalar_t value)
        {
                m_value += std::exp(value * m_beta);
        }

        void softmax_criterion_t::accumulate(const vector_t& vgrad, const scalar_t value)
        {
                m_value += std::exp(value * m_beta);
                m_vgrad += vgrad * std::exp(value * m_beta);
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

                return std::log(m_value) / m_beta;
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
