#include <set>
#include <iostream>
#include "math/clamp.hpp"
#include "text/cmdline.h"

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

int main(int argc, const char* argv[])
{
        // parse the command line
        nano::cmdline_t cmdline("compute all convolution networks with squared kernels for a given input size");
        cmdline.add("", "irows",        "number of input rows [16, 256]");
        cmdline.add("", "icols",        "number of input columns [16, 256]");
        cmdline.add("", "max-krows",    "maximum convolution size [3, 15]");
        cmdline.add("", "pooling",      "use pooling after each layer");

        cmdline.process(argc, argv);

        // check arguments and options
        const auto irows = nano::clamp(cmdline.get<int>("irows"), 16, 256);
        const auto icols = nano::clamp(cmdline.get<int>("icols"), 16, 256);
        const auto max_krows = nano::clamp(cmdline.get<int>("max-krows"), 3, 15);
        const auto min_krows = 3;
        const auto use_pooling = cmdline.has("pooling");

        const convnets_t nets = make_convnets(irows, icols, min_krows, max_krows, use_pooling, {});

        for (const auto& net : nets)
        {
                print(irows, icols, use_pooling, net);
        }

        // OK
        return EXIT_SUCCESS;
}
