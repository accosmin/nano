#pragma once

#include "libnanocv/arch.h"
#include "libnanocv/string.h"
#include "libnanocv/optimizer.h"

namespace ncv
{
        typedef std::pair<vector_t, scalar_t>           solution_t;
        typedef std::vector<solution_t>                 solutions_t;

        ///
        /// \brief test optimization problem
        ///
        struct NANOCV_PUBLIC function_t
        {
                function_t(const string_t& name = string_t(),
                           const opt_opsize_t& opsize = opt_opsize_t(),
                           const opt_opfval_t& opfval = opt_opfval_t(),
                           const opt_opgrad_t& opgrad = opt_opgrad_t(),
                           const solutions_t& solutions = solutions_t())
                        :       m_name(name),
                                m_opsize(opsize),
                                m_opfval(opfval),
                                m_opgrad(opgrad),
                                m_solutions(solutions)
                {
                }

                // attributes
                string_t        m_name;

                opt_opsize_t    m_opsize;
                opt_opfval_t    m_opfval;
                opt_opgrad_t    m_opgrad;

                solutions_t     m_solutions;
        };
}
