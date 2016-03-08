#include "unit_test.hpp"
#include "math/gauss.hpp"

ZOB_BEGIN_MODULE(test_gauss)

ZOB_CASE(evaluate)
{
        using std::size_t;

        const std::vector<double> sigmas = { 0.2, 0.5, 0.7, 1.0, 1.5, 2.0, 2.5 };
        const std::vector<double> cutoffs = { 0.001, 0.01, 0.1 };

        const math::gauss::kernel_normalization normalize = math::gauss::kernel_normalization::on;

        // test various variances
        for (double sigma : sigmas)
        {
                // test various cutoffs (skip low values in the kernel)
                for (double cutoff : cutoffs)
                {
                        const auto kernel = math::gauss_kernel_t<double>(sigma, cutoff, normalize);

                        std::cout << "sigma = " << sigma << ", cutoff = " << cutoff << std::endl;
                        std::cout << "kernel = {";
                        for (size_t k = 0; k < kernel.size(); ++ k)
                        {
                                std::cout << kernel[k] << (k + 1 == kernel.size() ? "" : ", ");
                        }
                        std::cout << "}" << std::endl << std::endl;

                        /// \todo more tests!

                        // check kernel sum
                        const double sum = kernel.sum();
                        ZOB_CHECK_LESS(sum, 1.0 + 1e-8);
                        ZOB_CHECK_GREATER(sum, 1.0 - 1e-8);
                }
        }
}

ZOB_END_MODULE()
