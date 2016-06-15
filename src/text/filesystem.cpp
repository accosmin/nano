#include "filesystem.h"

namespace nano
{
        string_t filename(const string_t& path)
        {
                const auto pos_dir = path.find_last_of("/\\");

                if (pos_dir == string_t::npos)
                {
                        return path;
                }
                else
                {
                        return path.substr(pos_dir + 1);
                }
        }

        string_t extension(const string_t& path)
        {
                const auto pos_ext = path.find_last_of('.');

                if (pos_ext == string_t::npos)
                {
                        return string_t();
                }
                else
                {
                        return path.substr(pos_ext + 1);
                }
        }

        string_t stem(const string_t& path)
        {
                const auto pos_dir = path.find_last_of("/\\");
                const auto pos_ext = path.find_last_of('.');

                if (pos_dir == string_t::npos)
                {
                        if (pos_ext == string_t::npos)
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
                        if (pos_ext == string_t::npos)
                        {
                                return path.substr(pos_dir + 1);
                        }
                        else
                        {
                                return path.substr(pos_dir + 1, pos_ext - pos_dir - 1);
                        }
                }
        }

        string_t dirname(const string_t& path)
        {
                const auto pos_dir = path.find_last_of("/\\");

                if (pos_dir == string_t::npos)
                {
                        return "./";
                }
                else
                {
                        return path.substr(0, pos_dir + 1);
                }
        }
}

