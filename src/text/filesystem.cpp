#include "filesystem.h"

namespace nano
{
        std::string filename(const std::string& path)
        {
                const auto pos_dir = path.find_last_of("/\\");

                if (pos_dir == std::string::npos)
                {
                        return path;
                }
                else
                {
                        return path.substr(pos_dir + 1);
                }
        }

        std::string extension(const std::string& path)
        {
                const auto pos_ext = path.find_last_of('.');

                if (pos_ext == std::string::npos)
                {
                        return std::string();
                }
                else
                {
                        return path.substr(pos_ext + 1);
                }
        }

        std::string stem(const std::string& path)
        {
                const auto pos_dir = path.find_last_of("/\\");
                const auto pos_ext = path.find_last_of('.');

                if (pos_dir == std::string::npos)
                {
                        if (pos_ext == std::string::npos)
                        {
                                return path;
                        }
                        else
                        {
                                return path.substr(0, pos_ext);
                        }
                }
                else
                {
                        if (pos_ext == std::string::npos)
                        {
                                return path.substr(pos_dir + 1);
                        }
                        else
                        {
                                return path.substr(pos_dir + 1, pos_ext - pos_dir - 1);
                        }
                }
        }

        std::string dirname(const std::string& path)
        {
                const auto pos_dir = path.find_last_of("/\\");

                if (pos_dir == std::string::npos)
                {
                        return "./";
                }
                else
                {
                        return path.substr(0, pos_dir + 1);
                }
        }
}

