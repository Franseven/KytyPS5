#ifndef EMULATOR_SRC_GRAPHICS_SHADER_SHADEROFFLINE_H_
#define EMULATOR_SRC_GRAPHICS_SHADER_SHADEROFFLINE_H_

// The offline half of the shader path: find every shader a game's own files hold, and put one in
// front of the compiler as a draw would — the header's registers decoded into the hardware state,
// stand-in vertex tables and stand-in guest memory — with no guest running.
//
// tests/ShaderBatchTool.cpp (the batch translator) and PipelineCache::PrecompileGame (the offline
// precompiler) both build on this, so both translate the same shader and, through
// PipelineCache::ProgramCache, compute the same disk cache keys.
//
// Where the shaders are: the AGC header block in eboot.bin's data segment (every PS5 title has one,
// so this half knows no engine) and, when the game has them, its CSDR bundles.

#include "graphics/guest_gpu/hardwareContext.h"
#include "graphics/shader/shader.h"
#include "graphics/shader/shaderCompiler.h"

#include <array>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <span>
#include <string>
#include <vector>

namespace Libs::Graphics::Offline {

// One shader as a game's files hold it: the header blob (header_size bytes, self-relative, so it
// survives being copied) and its machine code.
struct ShaderBlob {
	std::vector<char> header;
	std::vector<char> code;
	std::string       source;
};

// Every shader the game's files hold, in the order the files lay them out. `report` gets one line
// per source that was read.
std::vector<ShaderBlob> EnumerateGame(const std::filesystem::path& game, std::string& report);

// A folder of <hash>.header + <hash>.code pairs: what the emulator's KYTY_SHADER_DUMP writes and
// what ShaderBatchTool reads. A game's files never hold all of its shaders, so this is the half
// that only a run of the game itself can supply.
std::vector<ShaderBlob> LoadCorpus(const std::filesystem::path& folder, std::string& report);

// The emulator's shader identity: XXH3_64bits over the code, as graphics/shader/shader.cpp
// computes it. Two blobs with the same hash are the same shader, however many containers carry it.
uint64_t ShaderHash(const ShaderBlob& blob);

// Stand-ins for the draw's vertex attribute and buffer tables: every semantic the header declares
// gets a 32-bit float attribute with its channel count, in a buffer of its own. The registers the
// header carries point at these arrays, so they must outlive the translation.
struct VertexTables {
	std::array<uint32_t, 256>    attrib {};
	std::array<uint32_t, 4 * 32> buffer {};
};

// A shader brought to the point where a draw would hand it to the compiler.
struct Program {
	ShaderParams           params;
	ShaderComputeInputInfo compute;
	ShaderPixelInputInfo   pixel;
	ShaderVertexInputInfo  vertex;
	VertexTables           vertex_tables;
	// The stage input the translation reads: it points into this struct, so a Program must not be
	// copied or moved once prepared.
	ShaderStageInputInfo input;
	ShaderType           stage          = ShaderType::Unknown;
	uint32_t             wave_size      = 64;
	uint32_t             user_data_base = 0;
};

enum class Status {
	Ok,
	// A fetch shader or a fused half: there is nothing to compile on its own.
	NotStandalone,
	// AgcCreateShader or the stand-in vertex tables refused it.
	Failed,
};

// Decodes the header into the hardware state and prepares the program the way the draw path does.
// `note` gets the stand-in tables' line, `failure` why a Failed shader failed.
Status PrepareOfflineProgram(const ShaderBlob& blob, Program& program, std::string& note,
                            std::string& failure);

// One 16-byte pattern laid over every address and over the user SGPRs. Read as a buffer descriptor
// (V#, 16-byte aligned) it is a raw 2 GB buffer, as an image descriptor (T#, 32-byte aligned) a null
// image (dword 0 is zero), as a pointer an address below 2^48 that reads the same pattern.
// Materialization then succeeds with every descriptor unbound, as when a game draws with nothing
// bound to a slot.
bool ReadPatternMemory(void* userdata, uint64_t address, std::span<uint32_t> values);
void FillUserData(HW::UserSgprInfo& user_data);

// A copy of `bytes` at an address the guest-visible blob can point into. Kept for the run: the
// shader map and the hash cache hold these addresses.
void* CopyAligned(const std::vector<char>& bytes, size_t alignment);

// Joins `text` onto `note` as the batch tool's RESULT note column does.
void AppendNote(std::string& note, const std::string& text);

} // namespace Libs::Graphics::Offline

#endif /* EMULATOR_SRC_GRAPHICS_SHADER_SHADEROFFLINE_H_ */
