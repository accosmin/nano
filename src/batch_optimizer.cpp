#include <mutex>
#include "batch/gd.h"
#include "batch/cgd.h"
#include "batch/lbfgs.h"

using namespace nano;

batch_optimizer_factory_t& nano::get_batch_optimizers()
{
        static batch_optimizer_factory_t manager;

        static std::once_flag flag;
        std::call_once(flag, [&m = manager] ()
        {
                m.add<batch_gd_t>("gd", "gradient descent");
                m.add<batch_cgd_prp_t>("cgd", "nonlinear conjugate gradient descent (default)");
                m.add<batch_cgd_n_t>("cgd-n", "nonlinear conjugate gradient descent (N)");
                m.add<batch_cgd_hs_t>("cgd-hs", "nonlinear conjugate gradient descent (HS)");
                m.add<batch_cgd_fr_t>("cgd-fr", "nonlinear conjugate gradient descent (FR)");
                m.add<batch_cgd_prp_t>("cgd-prp", "nonlinear conjugate gradient descent (PRP+)");
                m.add<batch_cgd_cd_t>("cgd-cd", "nonlinear conjugate gradient descent (CD)");
                m.add<batch_cgd_ls_t>("cgd-ls", "nonlinear conjugate gradient descent (LS)");
                m.add<batch_cgd_dy_t>("cgd-dy", "nonlinear conjugate gradient descent (DY)");
                m.add<batch_cgd_dycd_t>("cgd-dycd", "nonlinear conjugate gradient descent (DYCD)");
                m.add<batch_cgd_dyhs_t>("cgd-dyhs", "nonlinear conjugate gradient descent (DYHS)");
                m.add<batch_lbfgs_t>("lbfgs", "limited-memory BFGS");
        });

        return manager;
}
