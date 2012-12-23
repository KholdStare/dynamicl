#define BOOST_TEST_DYN_LINK
#define BOOST_TEST_MAIN
#include <boost/test/unit_test.hpp>

#include "utils.h"

using namespace DynamiCL;

BOOST_AUTO_TEST_SUITE(utils)

BOOST_AUTO_TEST_CASE( stripExtension_test )
{
    using namespace std;

    BOOST_CHECK_EQUAL( "", stripExtension("") );
    BOOST_CHECK_EQUAL( "a", stripExtension("a") );
    BOOST_CHECK_EQUAL( "longer", stripExtension("longer") );
    BOOST_CHECK_EQUAL( "", stripExtension(".") );
    BOOST_CHECK_EQUAL( "", stripExtension(".jpg") );
    BOOST_CHECK_EQUAL( "hello", stripExtension("hello.jpg") );
    BOOST_CHECK_EQUAL( ".jpg", stripExtension(".jpg.bmp") );
    BOOST_CHECK_EQUAL( "hello.jpg", stripExtension("hello.jpg.bmp") );
}

BOOST_AUTO_TEST_SUITE_END()
