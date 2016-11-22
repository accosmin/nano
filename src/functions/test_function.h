#pragma once

#include "function.h"

namespace nano
{
        ///
        /// \brief generic multi-dimensional function used for testing and benchmarking.
        /// NB: the stochastic approximation is disabled.
        /// NB: need to only define ::vgrad(x, &gx).
        ///
        struct test_function_t : public function_t
        {
                explicit test_function_t(const char* name,
                        const tensor_size_t size, const tensor_size_t min_size, const tensor_size_t max_size,
                        const convexity convex,
                        const scalar_t domain) :
                        m_name(name),
                        m_size(size), m_min_size(min_size), m_max_size(max_size),
                        m_convex(convex),
                        m_domain(domain)
                {
                }

                string_t name() const override
                {
                        return string_t(m_name) + "[" + std::to_string(size()) + "D]";
                }

                bool is_valid(const vector_t& x) const override
                {
                        return x.lpNorm<Eigen::Infinity>() < m_domain;
                }

                bool is_convex() const override
                {
                        return m_convex != convexity::no;
                }

                tensor_size_t size() const override
                {
                        return m_size;
                }

                tensor_size_t min_size() const override
                {
                        return m_min_size;
                }

                tensor_size_t max_size() const override
                {
                        return m_max_size;
                }

                size_t stoch_ratio() const override
                {
                        // no stochastic approximation
                        return 1;
                }

                void stoch_next() const override
                {
                        // no stochastic approximation
                }

        protected:

                scalar_t stoch_vgrad(const vector_t& x, vector_t* gx) const override
                {
                        // no stochastic approximation
                        return vgrad(x, gx);
                }

        private:

                // attributes
                const char*     m_name;
                tensor_size_t   m_size, m_min_size, m_max_size;
                convexity       m_convex;
                scalar_t        m_domain;                       ///< domain = hyper-ball{0, m_domain}
        };
}
