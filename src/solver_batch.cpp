#include <mutex>
#include "batch/solver_batch_gd.h"
#include "batch/solver_batch_cgd.h"
#include "batch/solver_batch_nag.h"
#include "batch/solver_batch_lbfgs.h"

using namespace nano;

batch_solver_factory_t& nano::get_batch_solvers()
{
        static batch_solver_factory_t manager;

        static std::once_flag flag;
        std::call_once(flag, [] ()
        {
                manager.add<batch_gd_t>("gd", "gradient descent");
                manager.add<batch_nag_t>("nag", "Nesterov's accelerated gradient");
                manager.add<batch_nagfr_t>("nagfr", "Nesterov's accelerated gradient with function value restarts");
                manager.add<batch_naggr_t>("naggr", "Nesterov's accelerated gradient with gradient restarts");
                manager.add<batch_cgd_prp_t>("cgd", "nonlinear conjugate gradient descent (default)");
                manager.add<batch_cgd_n_t>("cgd-n", "nonlinear conjugate gradient descent (N)");
                manager.add<batch_cgd_hs_t>("cgd-hs", "nonlinear conjugate gradient descent (HS)");
                manager.add<batch_cgd_fr_t>("cgd-fr", "nonlinear conjugate gradient descent (FR)");
                manager.add<batch_cgd_prp_t>("cgd-prp", "nonlinear conjugate gradient descent (PRP+)");
                manager.add<batch_cgd_cd_t>("cgd-cd", "nonlinear conjugate gradient descent (CD)");
                manager.add<batch_cgd_ls_t>("cgd-ls", "nonlinear conjugate gradient descent (LS)");
                manager.add<batch_cgd_dy_t>("cgd-dy", "nonlinear conjugate gradient descent (DY)");
                manager.add<batch_cgd_dycd_t>("cgd-dycd", "nonlinear conjugate gradient descent (DYCD)");
                manager.add<batch_cgd_dyhs_t>("cgd-dyhs", "nonlinear conjugate gradient descent (DYHS)");
                manager.add<batch_lbfgs_t>("lbfgs", "limited-memory BFGS");
        });

        return manager;
}
