// Offline batch translation of guest shaders extracted from the game files: <hash>.header (the
// AGC header, pointers self-relative), <hash>.code and manifest.csv in one folder.
//
// Each shader takes the path a draw or dispatch takes in-game: AgcCreateShader, the header's
// registers decoded into the hardware state as the PM4 handlers do, PrepareProgram, then
// TranslateProgram and ExtractResourcePlan with ProgramCache::Get's options (non-fatal, bindless
// images). Draw-time state the header does not hold gets defaults: the input primitive that fits
// the header's GE_CNTL group size (points, lines or triangles), identity pixel interpolators, the
// default export mapping, and stand-in vertex attribute and buffer tables built from the
// header's own input semantics. The host subgroup size is 32, as on
// NVIDIA. Shader function calls are expanded first, as PipelineCache::ExpandShaderFunctions does
// before translating in game: a callee's address is the code base plus an S_GETPC_B64-derived
// offset, so it lands inside the shader's own code and the reads are served from the buffer.
//
// usage: shader_batch_tool <shader folder> <log file> [first manifest row] [end row]
//        shader_batch_tool --enumerate <game folder> <log file> [first] [end]
// Prints "START <row> <hash>" before and one tab-separated RESULT line after each shader. A
// fatal error exits with code 321 after START, so a driver can resume at the next row. The
// RESULT line lists every instruction the decoder does not support, not only the first one the
// CFG stops at. KYTY_BATCH_DUMP_DIR makes rejected shaders dump their IR (see main).
//
// --enumerate finds the shaders in the game's own files instead of taking a folder someone
// extracted: the AGC headers in eboot.bin's data segment (every PS5 title has one, so this needs
// no knowledge of the engine) and, when the game has them, its CSDR bundles. The two sources are
// the extraction half of tools/local/static-precompile/agc.py.

#include "common/assert.h"
#include "common/emulatorConfig.h"
#include "common/logging/log.h"
#include "common/subsystems.h"
#include "common/threads.h"
#include "graphics/guest_gpu/gpu_defs.h"
#include "graphics/guest_gpu/hardwareContext.h"
#include "graphics/guest_gpu/pm4.h"
#include "graphics/shader/recompiler/ShaderRecompiler.h"
#include "graphics/shader/recompiler/frontend/decode/ShaderDecoder.h"
#include "graphics/shader/recompiler/frontend/decode/ShaderFunctions.h"
#include "graphics/shader/shader.h"
#include "graphics/shader/shaderCompiler.h"
#include "graphics/shader/shaderVertexMetadata.h"
#include "libs/agc.h"
#include "spirv-tools/libspirv.hpp"

#include <algorithm>
#include <array>
#include <chrono>
#include <cinttypes>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fmt/format.h>
#include <fstream>
#include <iterator>
#include <malloc.h>
#include <map>
#include <optional>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>
#include <xxhash.h>

#ifdef _WIN32
#define NOMINMAX
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#endif

#include "graphics/shader/ShaderOffline.h"
namespace Libs::Graphics {
namespace {

#ifdef _WIN32
// A crash (not a fatal error, which exits with 321) would lose the buffered log. Print the
// fault and the host stack on one CRASH line before the process dies, so the driver can say
// where the shader crashed. C++ exceptions, which the recompiler uses internally, pass through.
LONG WINAPI ReportCrash(EXCEPTION_POINTERS* info) {
	const auto code = info->ExceptionRecord->ExceptionCode;
	if (code != EXCEPTION_ACCESS_VIOLATION && code != EXCEPTION_STACK_OVERFLOW &&
	    code != EXCEPTION_ILLEGAL_INSTRUCTION && code != EXCEPTION_INT_DIVIDE_BY_ZERO &&
	    code != EXCEPTION_PRIV_INSTRUCTION && code != EXCEPTION_ARRAY_BOUNDS_EXCEEDED) {
		return EXCEPTION_CONTINUE_SEARCH;
	}
	auto stack = Common::HostBacktrace();
	std::replace(stack.begin(), stack.end(), '\n', '|');
	std::printf("CRASH\t0x%08lx\t0x%016llx\t%s\n", static_cast<unsigned long>(code),
	            static_cast<unsigned long long>(
	                reinterpret_cast<uintptr_t>(info->ExceptionRecord->ExceptionAddress)),
	            stack.c_str());
	std::fflush(stdout);
	return EXCEPTION_CONTINUE_SEARCH;
}
#endif


struct ManifestRow {
	uint64_t    hash = 0;
	std::string type;
};

std::vector<ManifestRow> ReadManifest(const std::filesystem::path& folder) {
	std::ifstream            file(folder / "manifest.csv");
	std::vector<ManifestRow> rows;
	std::string              line;
	// The dump ShaderDumpConsider writes has no header row, the one a hand-made folder carries does:
	// tell them apart by the first field being a hash, or the header line would drop a shader.
	const auto is_hash = [](std::string_view field) {
		return field.size() == 16 &&
		       field.find_first_not_of("0123456789abcdefABCDEF") == std::string_view::npos;
	};
	while (std::getline(file, line)) {
		const auto first  = line.find(',');
		const auto second = line.find(',', first + 1);
		if (line.empty() || first == std::string::npos || second == std::string::npos ||
		    !is_hash(std::string_view(line).substr(0, first))) {
			continue;
		}
		rows.push_back({.hash = std::strtoull(line.substr(0, first).c_str(), nullptr, 16),
		                .type = line.substr(first + 1, second - first - 1)});
	}
	return rows;
}

std::vector<char> ReadFile(const std::filesystem::path& path) {
	std::ifstream file(path, std::ios::binary);
	return {std::istreambuf_iterator<char>(file), std::istreambuf_iterator<char>()};
}

std::string ValidateSpirv(const std::vector<uint32_t>& spirv) {
	spvtools::SpirvTools tools(SPV_ENV_VULKAN_1_3);
	std::string          first;
	tools.SetMessageConsumer(
	    [&first](spv_message_level_t, const char*, const spv_position_t&, const char* message) {
		    if (first.empty()) {
			    first = message;
		    }
	    });
	if (tools.Validate(spirv)) {
		return "valid";
	}
	std::replace(first.begin(), first.end(), '\t', ' ');
	std::replace(first.begin(), first.end(), '\n', ' ');
	return "invalid: " + first.substr(0, 300);
}

// With KYTY_BATCH_DUMP_DIR set, a shader whose SPIR-V fails validation leaves
// <folder>/invalid/<hash>.spv, its disassembly (.spvasm) and its IR (.ir) behind. With
// KYTY_BATCH_DUMP_ALL set as well, every compiled shader leaves them in <folder>/all, with its
// decoded code (.asm).
void DumpSpirv(const char* subfolder, const std::string& name, const std::vector<uint32_t>& spirv,
               const std::string& ir_dump, const std::string& decoded_dump = {}) {
	const char* folder = std::getenv("KYTY_BATCH_DUMP_DIR");
	if (folder == nullptr) {
		return;
	}
	const auto base = std::filesystem::path(folder) / subfolder;
	std::filesystem::create_directories(base);
	std::ofstream(base / (name + ".spv"), std::ios::binary)
	    .write(reinterpret_cast<const char*>(spirv.data()),
	           static_cast<std::streamsize>(spirv.size() * sizeof(uint32_t)));
	// KYTY_BATCH_DUMP_SPV_ONLY: the module alone, for runs over thousands of shaders.
	if (std::getenv("KYTY_BATCH_DUMP_SPV_ONLY") != nullptr) {
		return;
	}
	std::string source;
	spvtools::SpirvTools(SPV_ENV_VULKAN_1_3)
	    .Disassemble(spirv, &source, SPV_BINARY_TO_TEXT_OPTION_FRIENDLY_NAMES);
	std::ofstream(base / (name + ".spvasm")) << source;
	if (!ir_dump.empty()) {
		std::ofstream(base / (name + ".ir")) << ir_dump;
	}
	if (!decoded_dump.empty()) {
		std::ofstream(base / (name + ".asm")) << decoded_dump;
	}
}

void Phase(const char* name) {
	std::printf("PHASE\t%s\n", name);
	std::fflush(stdout);
}

double MillisecondsSince(std::chrono::steady_clock::time_point begin) {
	return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - begin)
	    .count();
}


std::string UnsupportedInstructions(std::span<const uint32_t> code) {
	ShaderRecompiler::Decoder::Program decoded;
	ShaderRecompiler::Decoder::DecodeProgram(code, decoded);
	std::vector<std::string> found;
	for (const auto& inst: decoded.instructions) {
		if (inst.opcode != ShaderRecompiler::Decoder::Opcode::UNSUPPORTED) {
			continue;
		}
		auto text = ShaderRecompiler::Decoder::InstructionToString(inst);
		if (const auto colon = text.find(": "); colon != std::string::npos) {
			text.erase(0, colon + 2);
		}
		while (!text.empty() && text.back() == ' ') {
			text.pop_back();
		}
		if (std::find(found.begin(), found.end(), text) == found.end()) {
			found.push_back(std::move(text));
		}
	}
	std::string joined;
	for (const auto& text: found) {
		Offline::AppendNote(joined, text);
	}
	return joined;
}


// One shader of the folder, translated exactly as a draw translates it: the shared offline unit
// brings the header to the point where the draw would hand it to the compiler, and from there this
// is the in-game path, with every descriptor unbound.
void RunShader(const std::vector<char>& header_bytes, const std::vector<char>& code_bytes,
               uint64_t hash, const std::string& type, const std::string& source) {
	const auto name = fmt::format("{:016x}", hash);

	Offline::ShaderBlob blob;
	blob.header = header_bytes;
	blob.code   = code_bytes;
	Offline::Program program;
	std::string      note;
	std::string      failure;

	const auto prepare_begin = std::chrono::steady_clock::now();
	const auto status        = Offline::PrepareOfflineProgram(blob, program, note, failure);
	if (status != Offline::Status::Ok) {
		// A fetch shader or a fused half, or a header the offline stand-in state cannot carry: one
		// line for both, since neither leaves anything to compile.
		const bool not_standalone = status == Offline::Status::NotStandalone;
		std::printf("RESULT\t%s\t%s\t-\t%s\t0\t0\t0\t0\t%s\t\n", name.c_str(), type.c_str(),
		            not_standalone ? "not standalone" : "failed",
		            not_standalone ? "fetch shader or fused half" : failure.c_str());
		return;
	}
	Offline::AppendNote(note, source);

	auto&      params         = program.params;
	const auto stage          = program.stage;
	const auto stage_input    = program.input;
	const auto wave_size      = program.wave_size;
	const auto user_data_base = program.user_data_base;
	const auto prepare_ms = MillisecondsSince(prepare_begin);
	if (params.hash != hash) {
		Offline::AppendNote(note, fmt::format("declared hash {:016x}", params.hash));
	}

	const char* stage_name = stage == ShaderType::Compute ? "CS"
	                         : stage == ShaderType::Pixel ? "PS"
	                         : stage == ShaderType::Mesh  ? "MS"
	                                                      : "VS";
	ShaderRecompiler::CompileOptions options;
	options.stage           = stage;
	options.shader_hash     = params.hash;
	options.user_data       = std::span<const uint32_t>(params.user_data).first(params.user_data_count);
	options.back_code       = params.back_code;
	// KYTY_BATCH_DUMP_IR: keep each shader's final IR, for the .ir of an invalid-SPIR-V dump.
	options.dump_ir         = std::getenv("KYTY_BATCH_DUMP_IR") != nullptr ||
	                  std::getenv("KYTY_BATCH_DUMP_ALL") != nullptr;
	options.early_dump      = false;
	options.dump_label      = "ShaderBatch";
	options.input_info      = stage_input;
	options.user_data_base  = user_data_base;
	options.wave_size       = wave_size;
	options.non_fatal       = true;
	options.bindless_images = true;

	// Function calls first, as PipelineCache::ExpandShaderFunctions does before translating (or the
	// raw s_swappc_b64 reaches the decoder, which does not implement it, and a shader the emulator
	// compiles in game is reported here as unsupported). A callee's address is the code base plus an
	// S_GETPC_B64-derived offset, so it lands inside `code` -- which here is a host copy, not a guest
	// address, so those reads are served from the buffer itself; anything else is stand-in memory.
	std::vector<uint32_t> expanded;
	{
		const auto  base  = params.Base();
		const auto  words = params.code;
		std::string reason;
		ShaderRecompiler::Decoder::ShaderFunctionExpander expander;
		if (!expander.Expand(words, base, options.user_data,
		                     [base, words](uint64_t address, std::span<uint32_t> out) {
			                     const auto bytes =
			                         static_cast<uint64_t>(words.size()) * sizeof(uint32_t);
			                     const auto want =
			                         static_cast<uint64_t>(out.size()) * sizeof(uint32_t);
			                     if (address >= base &&
			                         (address - base) % sizeof(uint32_t) == 0 &&
			                         address - base + want <= bytes) {
				                     std::memcpy(
				                         out.data(),
				                         words.data() + (address - base) / sizeof(uint32_t), want);
				                     return true;
			                     }
			                     return Offline::ReadPatternMemory(nullptr, address, out);
		                     },
		                     expanded, reason, options.wave_size)) {
			Offline::AppendNote(note, "expansion: " + reason);
		} else if (!expanded.empty()) {
			// Silence here used to mean two different things: "no calls at all" and "the calls were
			// expanded". Name the second, or a report cannot tell which shaders the pass rescued.
			Offline::AppendNote(note, fmt::format("calls expanded to {} words", expanded.size()));
		}
	}
	const std::span<const uint32_t> translate_code =
	    expanded.empty() ? params.code : std::span<const uint32_t>(expanded);
	// The list the report shows has to be the code that was translated: with the calls expanded the
	// raw s_swappc_b64 is gone, and naming it would call unsupported a shader that compiled.
	const auto unsupported     = UnsupportedInstructions(translate_code);
	const auto translate_begin = std::chrono::steady_clock::now();
	auto       translated      = ShaderRecompiler::TranslateProgram(translate_code, options);
	const auto translate_ms    = MillisecondsSince(translate_begin);
	// Upstream executes BVH dispatches now (337f4320): no "ray tracing" outcome any more.
	const char* outcome        = translated.unsupported ? "gave up" : "ok";
	const bool   ok         = !translated.unsupported;
	const bool   dispatcher = ok && translated.program.dispatcher_fallback;
	const size_t blocks     = ok ? translated.program.blocks.size() : 0;
	double       plan_ms    = 0;
	double       emit_ms    = 0;
	std::string  spirv      = "-";
	size_t       spirv_words = 0;
	if (ok) {
		// The rest of the in-game path, with every descriptor unbound: materialize the resource
		// plan, emit SPIR-V and validate it.
		const auto plan_begin = std::chrono::steady_clock::now();
		const auto plan       = ShaderRecompiler::IR::ExtractResourcePlan(translated.program);
		plan_ms               = MillisecondsSince(plan_begin);
		Phase("materialize");
		ShaderRecompiler::IR::ResourceSnapshot       resources;
		ShaderRecompiler::IR::ResourceSpecialization specialization;
		const ShaderRecompiler::IR::SrtRuntime       runtime {
		          .user_data                  = options.user_data,
		          .shader_base                = reinterpret_cast<uint64_t>(params.code.data()),
		          .read_memory                = Offline::ReadPatternMemory,
		          .userdata                   = nullptr,
		          .read_specialization_memory = Offline::ReadPatternMemory,
        };
		if (!ShaderRecompiler::IR::MaterializeResources(plan, runtime, resources, specialization)) {
			spirv = fmt::format("materialization failed (line {})",
			                    ShaderRecompiler::IR::LastIndirectImageFailureLine());
		} else {
			Phase("emit");
			const auto emit_begin = std::chrono::steady_clock::now();
			const auto push_data  = stage == ShaderType::Mesh
			                            ? ShaderRecompiler::IR::PushData::MeshDrawDwordCount
			                            : 0u;
			auto portable = specialization;
			// Mirrors the live path (PipelineCache::ProgramCache::Get): with the mode on the module is
			// emitted with the buffer formats left open, so the same SPIR-V serves every draw that
			// differs only in them. Off, this changes nothing.
			ShaderRecompiler::IR::PortableFormats(plan.info, portable);
			auto compiled = ShaderRecompiler::CompileProgram(std::move(translated), options,
			                                                 portable, push_data);
			Phase("validate");
			spirv       = ValidateSpirv(compiled.spirv);
			spirv_words = compiled.spirv.size();
			emit_ms     = MillisecondsSince(emit_begin);
			// Portable modules spend a push-data dword per buffer whose stride or format they read at
			// run time (ShaderIR.h, BufferWord), so a shader that fit before can stop fitting
			// PushData::DwordCount and silently move to the storage-buffer path it would rather not
			// take. Only a note: the shader is still correct, and this is the only place that sees it.
			const auto& bindings = compiled.program.bindings;
			if (bindings.ShaderDataDwords() != 0 && !bindings.UsesPushData()) {
				Offline::AppendNote(note, fmt::format("shader data {} dwords does not fit push data",
				                             bindings.ShaderDataDwords()));
			}
			if (spirv != "valid") {
				DumpSpirv("invalid", name, compiled.spirv, compiled.ir_dump);
			}
			if (std::getenv("KYTY_BATCH_DUMP_ALL") != nullptr) {
				DumpSpirv("all", name, compiled.spirv, compiled.ir_dump, compiled.decoded_dump);
			}
		}
	}
	std::printf("RESULT\t%s\t%s\t%s\t%s\t%.2f\t%.2f\t%d\t%zu\t%s\t%s\t%s\t%zu\t%.2f\n",
	            name.c_str(), type.c_str(), stage_name, outcome, prepare_ms,
	            translate_ms + plan_ms, dispatcher ? 1 : 0, blocks, note.c_str(),
	            unsupported.c_str(), spirv.c_str(), spirv_words, emit_ms);
}
} // namespace
} // namespace Libs::Graphics

int main(int argc, char* argv[]) {
	using namespace Libs::Graphics;
	const bool enumerate = argc > 1 && std::strcmp(argv[1], "--enumerate") == 0;
	const int  base      = enumerate ? 2 : 1;
	if (argc < base + 2) {
		std::fprintf(stderr,
		             "usage: shader_batch_tool <shader folder> <log file> [first row] [end row]\n"
		             "       shader_batch_tool --enumerate <game folder> <log file> [first] [end]\n");
		return 2;
	}
	const std::filesystem::path input = argv[base];
	const size_t                first = argc > base + 2 ? std::strtoull(argv[base + 2], nullptr, 10) : 0;
	const size_t                end   = argc > base + 3 ? std::strtoull(argv[base + 3], nullptr, 10) : SIZE_MAX;

	static Common::Subsystems subsystems;
	Common::InitializeThreads();
	subsystems.Initialize<Config::Lifecycle>();
	Config::ConfigOptions config;
	config.printf_direction   = Config::LogDirection::File;
	config.printf_output_file = argv[base + 1];
	// KYTY_BATCH_DUMP_DIR=<folder>: the IR of every shader that gives up in resource tracking goes
	// to <folder>/rejected/<stage>_<hash>.ir, as with --graphics-debug-dump in the emulator.
	if (const char* dump = std::getenv("KYTY_BATCH_DUMP_DIR"); dump != nullptr) {
		config.graphics_debug_dump_enabled = true;
		config.shader_log_folder           = dump;
	}
	Config::Load(config);
	subsystems.Initialize<Log::Lifecycle>();
	ShaderInit();
#ifdef _WIN32
	AddVectoredExceptionHandler(1, ReportCrash);
#endif

	if (enumerate) {
		// The shaders a game's own files hold, translated exactly as the manifest path translates
		// a folder someone else extracted: hash = XXH3_64bits(code), the emulator's ShaderParams.h
		// hash (graphics/shader/shader.cpp), so a RESULT's "declared hash" note stays quiet.
		static constexpr std::array<const char*, 9> type_names = {"Cs",  "Ps",      "Gs",  "Hs",
		                                                          "Gf",  "Hf",      "Gb",  "Hb",
		                                                          "Fs"};
		std::string                                report;
		auto                                       shaders = Offline::EnumerateGame(input, report);
		std::printf("ENUM\t%s\n%s", input.string().c_str(), report.c_str());
		std::fflush(stdout);

		// KYTY_ENUM_DUMP_DIR=<folder>: leave the enumerated shaders as the folder the manifest path
		// (and the offline precompiler) reads, <hash>.header + <hash>.code + manifest.csv, one entry
		// per distinct shader.
		std::vector<size_t> distinct;
		std::map<uint64_t, size_t> seen;
		for (size_t i = 0; i < shaders.size(); i++) {
			const auto hash = XXH3_64bits(shaders[i].code.data(), shaders[i].code.size());
			if (seen.emplace(hash, i).second) {
				distinct.push_back(i);
			}
		}
		std::printf("DISTINCT\t%zu of %zu\n", distinct.size(), shaders.size());
		if (const char* dir = std::getenv("KYTY_ENUM_DUMP_DIR"); dir != nullptr) {
			const std::filesystem::path out = dir;
			std::filesystem::create_directories(out);
			std::ofstream manifest(out / "manifest.csv");
			manifest << "hash,type,source\n";
			for (const auto i: distinct) {
				const auto& blob = shaders[i];
				const auto  hash = XXH3_64bits(blob.code.data(), blob.code.size());
				const auto  name = fmt::format("{:016x}", hash);
				std::ofstream(out / (name + ".header"), std::ios::binary)
				    .write(blob.header.data(), static_cast<std::streamsize>(blob.header.size()));
				std::ofstream(out / (name + ".code"), std::ios::binary)
				    .write(blob.code.data(), static_cast<std::streamsize>(blob.code.size()));
				const auto kind = static_cast<uint8_t>(blob.header[90]);
				manifest << name << ',' << (kind < type_names.size() ? type_names[kind] : "??") << ','
				         << blob.source << '\n';
			}
			std::printf("DUMP\t%s\n", out.string().c_str());
			std::fflush(stdout);
		}

		for (size_t i = first; i < std::min(end, shaders.size()); i++) {
			const auto& blob = shaders[i];
			const auto  hash = XXH3_64bits(blob.code.data(), blob.code.size());
			const auto  kind = static_cast<uint8_t>(blob.header[90]);
			const auto  type = kind < type_names.size() ? type_names[kind] : "??";
			std::printf("START\t%zu\t%016" PRIx64 "\n", i, hash);
			std::fflush(stdout);
			RunShader(blob.header, blob.code, hash, type, blob.source);
			std::fflush(stdout);
		}
		std::printf("DONE\t%zu\n", shaders.size());
		return 0;
	}

	const auto rows = ReadManifest(input);
	for (size_t i = first; i < std::min(end, rows.size()); i++) {
		const auto name   = fmt::format("{:016x}", rows[i].hash);
		auto       header = ReadFile(input / (name + ".header"));
		auto       code   = ReadFile(input / (name + ".code"));
		std::printf("START\t%zu\t%016" PRIx64 "\n", i, rows[i].hash);
		std::fflush(stdout);
		RunShader(header, code, rows[i].hash, rows[i].type, name);
		std::fflush(stdout);
	}
	std::printf("DONE\t%zu\n", rows.size());
	return 0;
}
