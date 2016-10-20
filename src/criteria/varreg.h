#pragma once

#include "average.h"
#include "softmax.h"

namespace nano
{
        ///
        /// \brief variance- regularized loss.
        /// (e.g. penalize high variance across training samples),
        ///     like in EBBoost/VadaBoost: http://www.cs.columbia.edu/~jebara/papers/vadaboost.pdf
        ///
        template <typename tcriterion>
        class var_criterion_t final : public tcriterion
        {
        public:

                explicit var_criterion_t(const string_t& configuration = string_t());

                virtual rcriterion_t clone(const string_t& configuration) const override;
                virtual rcriterion_t clone() const override;

                virtual scalar_t value() const override;
                virtual vector_t vgrad() const override;
                virtual bool can_regularize() const override;

        protected:

                virtual void clear() override;
                virtual void accumulate(scalar_t value) override;
                virtual void accumulate(const vector_t& vgrad, scalar_t value) override;
                virtual void accumulate(const criterion_t& other) override;

        private:

                // attributes
                scalar_t        m_value2;        ///< cumulated squared loss value
                vector_t        m_vgrad2;        ///< cumulated loss value multiplied with the gradient
        };

        template <typename tcriterion>
        var_criterion_t<tcriterion>::var_criterion_t(const string_t& configuration) :
                tcriterion(configuration),
                m_value2(0)
        {
        }

        template <typename tcriterion>
        rcriterion_t var_criterion_t<tcriterion>::clone(const string_t& configuration) const
        {
                return std::make_unique<var_criterion_t<tcriterion>>(configuration);
        }

        template <typename tcriterion>
        rcriterion_t var_criterion_t<tcriterion>::clone() const
        {
                return std::make_unique<var_criterion_t<tcriterion>>(*this);
        }

        template <typename tcriterion>
        void var_criterion_t<tcriterion>::clear()
        {
                tcriterion::clear();

                m_value2 = 0;

                m_vgrad2.resize(this->psize());
                m_vgrad2.setZero();
        }

        template <typename tcriterion>
        void var_criterion_t<tcriterion>::accumulate(scalar_t value)
        {
                tcriterion::accumulate(value);

                m_value2 += value * value;
        }

        template <typename tcriterion>
        void var_criterion_t<tcriterion>::accumulate(const vector_t& vgrad, scalar_t value)
        {
                tcriterion::accumulate(vgrad, value);

                m_value2 += value * value;
                m_vgrad2 += value * vgrad;
        }

        template <typename tcriterion>
        void var_criterion_t<tcriterion>::accumulate(const criterion_t& other)
        {
                tcriterion::accumulate(other);

                assert(dynamic_cast<const var_criterion_t<tcriterion>*>(&other));
                const auto& vother = static_cast<const var_criterion_t<tcriterion>&>(other);
                m_value2 += vother.m_value2;
                m_vgrad2 += vother.m_vgrad2;
        }

        template <typename tcriterion>
        scalar_t var_criterion_t<tcriterion>::value() const
        {
                const scalar_t count = static_cast<scalar_t>(this->count());

                return  this->lweight() * (tcriterion::value()) +
                        this->rweight() * (count * m_value2 - tcriterion::m_value * tcriterion::m_value) / (count * count);
        }

        template <typename tcriterion>
        vector_t var_criterion_t<tcriterion>::vgrad() const
        {
                const scalar_t count = static_cast<scalar_t>(this->count());

                return  this->lweight() * (tcriterion::vgrad()) +
                        this->rweight() * (2 * (count * m_vgrad2 - tcriterion::m_value * tcriterion::m_vgrad) / (count * count));
        }

        template <typename tcriterion>
        bool var_criterion_t<tcriterion>::can_regularize() const
        {
                return true;
        }

        using average_var_criterion_t = var_criterion_t<average_criterion_t>;
        using softmax_var_criterion_t = var_criterion_t<softmax_criterion_t>;
}

