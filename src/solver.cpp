#include <mutex>
#include "solvers/solver_gd.h"
#include "solvers/solver_cgd.h"
#include "solvers/solver_lbfgs.h"

using namespace nano;

solver_factory_t& nano::get_solvers()
{
        static solver_factory_t manager;

        static std::once_flag flag;
        std::call_once(flag, [] ()
        {
                manager.add<solver_gd_t>("gd", "gradient descent");
                manager.add<solver_cgd_prp_t>("cgd", "nonlinear conjugate gradient descent (default)");
                manager.add<solver_cgd_n_t>("cgd-n", "nonlinear conjugate gradient descent (N)");
                manager.add<solver_cgd_hs_t>("cgd-hs", "nonlinear conjugate gradient descent (HS)");
                manager.add<solver_cgd_fr_t>("cgd-fr", "nonlinear conjugate gradient descent (FR)");
                manager.add<solver_cgd_prp_t>("cgd-prp", "nonlinear conjugate gradient descent (PRP+)");
                manager.add<solver_cgd_cd_t>("cgd-cd", "nonlinear conjugate gradient descent (CD)");
                manager.add<solver_cgd_ls_t>("cgd-ls", "nonlinear conjugate gradient descent (LS)");
                manager.add<solver_cgd_dy_t>("cgd-dy", "nonlinear conjugate gradient descent (DY)");
                manager.add<solver_cgd_dycd_t>("cgd-dycd", "nonlinear conjugate gradient descent (DYCD)");
                manager.add<solver_cgd_dyhs_t>("cgd-dyhs", "nonlinear conjugate gradient descent (DYHS)");
                manager.add<solver_lbfgs_t>("lbfgs", "limited-memory BFGS");
        });

        return manager;
}
