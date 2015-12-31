#include <vector>
#include <iostream>
#include "math/clamp.hpp"
#include <boost/program_options.hpp>

namespace
{
        using convnet_t = std::vector<int>;
        using convnets_t = std::set<convnet_t>;

        bool valid_layer(const int irows, const int icols, const int krows, const int kcols)
        {
                return irows >= krows && icols >= kcols;
        }

        int make_orows(const int irows, const int krows, const bool use_pooling)
        {
                const auto orows = irows - krows + 1;
                return use_pooling ? ((orows + 1) / 2) : orows;
        }

        int make_ocols(const int icols, const int kcols, const bool use_pooling)
        {
                return make_orows(icols, kcols, use_pooling);
        }

        convnet_t normalize(convnet_t net)
        {
                std::sort(net.begin(), net.end(), std::greater<int>());
                return net;
        }

        void print(int irows, int icols, const bool use_pooling, const convnet_t& net)
        {
                std::cout << irows << "x" << icols << " -> ";
                for (std::size_t i = 0; i < net.size(); ++ i)
                {
                        const int krows = net[i];
                        const int kcols = krows;
                        std::cout << "@" << krows << "x" << kcols << (use_pooling ? "p " : " ");
                        irows = make_orows(irows, krows, use_pooling);
                        icols = make_ocols(icols, kcols, use_pooling);
                }
                std::cout << "-> " << irows << "x" << icols << std::endl;
        }

        convnets_t make_convnets(const int irows, const int icols, const int min_krows, const int max_krows,
                const bool use_pooling, const convnet_t& basenet)
        {
                convnets_t nets;

                if (!basenet.empty() && (irows < min_krows || icols < min_krows))
                {
                        nets.insert(normalize(basenet));
                }

                for (int krows = min_krows; krows <= max_krows; krows += 2)
                {
                        const int kcols = krows;

                        if (valid_layer(irows, icols, krows, kcols))
                        {
                                convnet_t knet = basenet;
                                knet.push_back(krows);

                                const auto knets = make_convnets(
                                        make_orows(irows, krows, use_pooling),
                                        make_ocols(icols, kcols, use_pooling),
                                        min_krows, max_krows, use_pooling,
                                        knet);

                                nets.insert(knets.begin(), knets.end());
                        }
                }

                return nets;
        }
}

int main(int argc, char* argv[])
{
        // parse the command line
        boost::program_options::options_description po_desc("", 160);
        po_desc.add_options()("help,h",
                "compute all possible convolution networks having squared kernels for a given input size");
        po_desc.add_options()("irows",
                boost::program_options::value<int>(),
                "number of input rows [16, 256]");
        po_desc.add_options()("icols",
                boost::program_options::value<int>(),
                "number of input columns [16, 256]");
        po_desc.add_options()("max-krows",
                boost::program_options::value<int>(),
                "maximum convolution size [3, 15]");
        po_desc.add_options()("pooling",
                "use pooling after each layer");

        boost::program_options::variables_map po_vm;
        boost::program_options::store(
                boost::program_options::command_line_parser(argc, argv).options(po_desc).run(),
                po_vm);
        boost::program_options::notify(po_vm);

        // check arguments and options
        if (	po_vm.empty() ||
                po_vm.count("help"))
        {
                std::cout << po_desc;
                return EXIT_FAILURE;
        }

        const int irows = math::clamp(po_vm["irows"].as<int>(), 16, 256);
        const int icols = math::clamp(po_vm["icols"].as<int>(), 16, 256);
        const int max_krows = math::clamp(po_vm["max-krows"].as<int>(), 3, 15);
        const int min_krows = 3;
        const bool use_pooling = po_vm.count("pooling");

        const convnets_t nets = make_convnets(irows, icols, min_krows, max_krows, use_pooling, {});

        for (const auto& net : nets)
        {
                print(irows, icols, use_pooling, net);
        }

        // OK
        return EXIT_SUCCESS;
}
