#pragma once

#include "state.h"
#include "function.h"

namespace nano
{
        ///
        /// \brief line-search (scalar) step.
        /// NB: using the notation from CG_DESCENT papers
        ///
        class ls_step_t
        {
        public:

                ///
                /// \brief constructor
                ///
                ls_step_t(const function_t& function, const state_t& state);

                ///
                /// \brief minimum allowed line-search step
                ///
                static scalar_t minimum();

                ///
                /// \brief maximum allowed line-search step
                ///
                static scalar_t maximum();

                ///
                /// \brief change the line-search step
                ///
                bool update(const scalar_t alpha);

                ///
                /// \brief check if the current step satisfies the Armijo condition (sufficient decrease)
                ///
                bool has_armijo(const scalar_t c1) const;

                ///
                /// \brief check if the current step satisfies the Wolfe condition (sufficient curvature)
                ///
                bool has_wolfe(const scalar_t c2) const;

                ///
                /// \brief check if the current step satisfies the strong Wolfe condition (sufficient curvature)
                ///
                bool has_strong_wolfe(const scalar_t c2) const;

                ///
                /// \brief check if the current step satisfies the approximate Wolfe condition (sufficient curvature)
                ///     see CG_DESCENT
                ///
                bool has_approx_wolfe(const scalar_t c1, const scalar_t c2, const scalar_t epsilon) const;

                ///
                /// \brief current step
                ///
                scalar_t alpha() const;

                ///
                /// \brief current function value
                ///
                scalar_t phi() const;

                ///
                /// \brief approximate function value (see CG_DESCENT)
                ///
                scalar_t approx_phi(const scalar_t epsilon) const;

                ///
                /// \brief initial function value
                ///
                scalar_t phi0() const;

                ///
                /// \brief current line-search function gradient
                ///
                scalar_t gphi() const;

                ///
                /// \brief initial line-search function gradient
                ///
                scalar_t gphi0() const;

                ///
                /// \brief currrent function value
                ///
                scalar_t func() const;

                ///
                /// \brief current gradient
                ///
                const vector_t& grad() const;

                ///
                /// \brief check if valid step
                ///
                operator bool() const;

        private:

                // attributes
                ref_function_t  m_function;
                ref_state_t     m_state;        ///< starting state for line-search
                scalar_t        m_gphi0;

                scalar_t        m_alpha;       ///< line-search step (current estimate)
                scalar_t        m_func;        ///< function value at alpha
                vector_t        m_grad;        ///< function gradient at alpha
                scalar_t        m_gphi;        ///< line-search function gradient at alpha
        };

        ///
        /// \brief compare two line-search steps (based on the function value)
        ///
        inline bool operator<(const ls_step_t& step1, const ls_step_t& step2)
        {
                return step1.phi() < step2.phi();
        }

}

