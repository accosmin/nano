#include "utest.h"
#include "vision/gauss.h"

NANO_BEGIN_MODULE(test_gauss)

NANO_CASE(evaluate)
{
        using namespace nano;

        const auto sigmas = { 0.2, 0.5, 0.7, 1.0, 1.5, 2.0, 2.5 };
        const auto cutoffs = { 0.001, 0.01, 0.1 };
        const auto normalize = nano::gauss::kernel_normalization::on;

        // test various variances
        for (auto sigma : sigmas)
        {
                // test various cutoffs (skip low values in the kernel)
                for (auto cutoff : cutoffs)
                {
                        const auto kernel = nano::make_gauss_kernel(
                                static_cast<scalar_t>(sigma),
                                static_cast<scalar_t>(cutoff),
                                normalize);

                        std::cout << "sigma = " << sigma << ", cutoff = " << cutoff << std::endl;
                        std::cout << "kernel = {" << kernel.transpose() << "}" << std::endl;

                        /// \todo more tests!

                        // check kernel sum
                        const double sum = kernel.sum();
                        NANO_CHECK_LESS(sum, 1.0 + 1e-6);
                        NANO_CHECK_GREATER(sum, 1.0 - 1e-6);
                }
        }
}

NANO_END_MODULE()
