#include "utils.h"

namespace DynamiCL
{

std::string stripExtension(std::string const& path)
{
    return path.substr(0, path.rfind('.'));
}

}
