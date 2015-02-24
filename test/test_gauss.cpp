#define BOOST_TEST_DYN_LINK
#define BOOST_TEST_MODULE "test_gauss"

#include <boost/test/unit_test.hpp>
#include "libnanocv/util/gauss.hpp"
#include <iostream>

BOOST_AUTO_TEST_CASE(test_gauss)
{
        using namespace ncv;

        using std::size_t;

        const std::vector<double> sigmas = { 0.2, 0.5, 0.7, 1.0, 1.5, 2.0, 2.5 };
        const std::vector<double> cutoffs = { 0.001, 0.01, 0.1 };

        const gauss::kernel_normalization normalize = gauss::kernel_normalization::on;

        // test various variances
        for (double sigma : sigmas)
        {
                // test various cutoffs (skip low values in the kernel)
                for (double cutoff : cutoffs)
                {
                        const auto kernel = gauss_kernel_t<double>(sigma, cutoff, normalize);

                        std::cout << "sigma = " << sigma << ", cutoff = " << cutoff << std::endl;
                        std::cout << "kernel = {";
                        for (size_t k = 0; k < kernel.size(); k ++)
                        {
                                std::cout << kernel[k] << (k + 1 == kernel.size() ? "" : ", ");
                        }
                        std::cout << "}" << std::endl << std::endl;

                        /// \todo more tests!

                        // check kernel sum
                        const double sum = kernel.sum();
                        BOOST_CHECK_LE(sum, 1.0 + 1e-8);
                        BOOST_CHECK_GE(sum, 1.0 - 1e-8);
                }
        }
}

