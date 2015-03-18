#pragma once

#include "../types.h"

namespace ncv
{
        typedef std::pair<ncv::vector_t, ncv::scalar_t>         solution_t;
        typedef std::vector<solution_t>                         solutions_t;

        ///
        /// \brief test optimization problem
        ///
        struct function_t
        {
                function_t(const ncv::string_t& name = ncv::string_t(),
                           const ncv::opt_opsize_t& opsize = ncv::opt_opsize_t(),
                           const ncv::opt_opfval_t& opfval = ncv::opt_opfval_t(),
                           const ncv::opt_opgrad_t& opgrad = ncv::opt_opgrad_t(),
                           const solutions_t& solutions = solutions_t())
                        :       m_name(name),
                                m_opsize(opsize),
                                m_opfval(opfval),
                                m_opgrad(opgrad),
                                m_solutions(solutions)
                {
                }

                // attributes
                ncv::string_t           m_name;

                ncv::opt_opsize_t       m_opsize;
                ncv::opt_opfval_t       m_opfval;
                ncv::opt_opgrad_t       m_opgrad;

                solutions_t             m_solutions;
        };
}
