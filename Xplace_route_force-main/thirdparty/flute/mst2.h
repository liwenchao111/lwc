#ifndef _MST2_H_
#define _MST2_H_

#include "global.h"

namespace Flute {

void  mst2_package_init( long  n );
void  mst2_package_done();
void  mst2( long n, Point* pt, long* parent );

}  // namespace Flute

#endif 

