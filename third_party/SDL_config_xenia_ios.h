/*
  Xenia iOS SDL override:
  Keep SDL on dummy-only video and sensor backends to avoid linking
  UIKit/CoreMotion implementation units in this static build.
*/

#include "SDL2/include/SDL_config_iphoneos.h"

#undef SDL_VIDEO_DRIVER_UIKIT
#undef SDL_SENSOR_COREMOTION
