#ifndef EMULATOR_INCLUDE_EMULATOR_GRAPHICS_WINDOW_H_
#define EMULATOR_INCLUDE_EMULATOR_GRAPHICS_WINDOW_H_

#include "common/abi.h"
#include "common/common.h"

namespace Libs::Graphics {

class Presenter;
class PipelineCache;

[[nodiscard]] Presenter& WindowInit(uint32_t width, uint32_t height);
void                     WindowRun();
void                     WindowShutdown();

// The pipeline cache of the window WindowInit created, or null before there is one. Emulator::Run
// seeds it offline (--precompile) with the device the game would have got.
PipelineCache* WindowPipelineCache();

// The offline precompiler runs no event loop, so its window would sit there unresponsive: ask for
// it hidden before the graphics subsystem creates it (Emulator::Run).
void WindowSetHidden();

} // namespace Libs::Graphics

#endif /* EMULATOR_INCLUDE_EMULATOR_GRAPHICS_WINDOW_H_ */
