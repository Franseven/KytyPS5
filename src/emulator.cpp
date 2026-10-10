#include "emulator.h"

#include "common/abi.h"
#include "common/assert.h"
#include "common/emulatorConfig.h"
#include "common/file.h"
#include "common/logging/log.h"
#include "common/profiler.h"
#include "common/singleton.h"
#include "common/stringUtils.h"
#include "common/subsystems.h"
#include "common/systemInfo.h"
#include "common/threads.h"
#include "graphics/host_gpu/renderer/pipeline/pipelineCache.h"
#include "graphics/presentation/window.h"
#include "graphics/shader/ShaderOffline.h"
#include "kernel/fileSystem.h"
#include "kernel/memory.h"
#include "kernel/pthread.h"
#include "kytyGitVersion.h"
#include "libs/agc.h"
#include "libs/audio.h"
#include "libs/controller.h"
#include "libs/libs.h"
#include "libs/network.h"
#include "loader/runtimeLinker.h"
#include "loader/systemContent.h"
#include "loader/timer.h"

#include <cstdlib>
#include <filesystem>
#include <iterator>
#include <map>
#include <thread>
#include <unordered_set>
#include <vector>

namespace Emulator {

static void PrintSystemInfo() {
	const Common::SystemInfo info = Common::GetSystemInfo();

#if defined(__APPLE__)
	static constexpr auto platform_name = "macOS";
#elif KYTY_PLATFORM == KYTY_PLATFORM_WINDOWS
	static constexpr auto platform_name = "Windows";
#elif KYTY_PLATFORM == KYTY_PLATFORM_LINUX
	static constexpr auto platform_name = "Linux";
#else
	static constexpr auto platform_name = "Unknown";
#endif

	LOGF("Build\n"
	     "  version: %s\n\n"
	     "Host\n"
	     "  os:      %s\n"
	     "  cpu:     %s\n"
	     "  threads: %u\n\n",
	     KYTY_BUILD_LABEL, platform_name, info.ProcessorName.c_str(),
	     std::thread::hardware_concurrency());
}

static void KytyClose() {
	auto* rt = Common::Singleton<Loader::RuntimeLinker>::Instance();

	rt->Clear();

	LOGF("done!\n");

	Common::Subsystems::EmergencyShutdownActive();
}

static void MountOrCreateDir(const std::filesystem::path& dir, const std::string& point) {
	if (!Common::File::IsDirectoryExisting(dir)) {
		Common::File::CreateDirectories(dir);
	}

	EXIT_NOT_IMPLEMENTED(!Common::File::IsDirectoryExisting(dir));

	Libs::LibKernel::FileSystem::Mount(dir, point);
	auto dir_text = Common::PathToString(dir);
	LOGF("Mounted %s -> %s\n", point.c_str(), dir_text.c_str());
}

static void MountSandboxDirs() {
	std::string title_id;
	if (!Loader::SystemContentParamSfoGetString("TITLE_ID", &title_id) || title_id.empty()) {
		title_id = "UNKNOWN";
	}

	MountOrCreateDir("_DownloadData/" + title_id, "/download0");
	MountOrCreateDir("_TempData/" + title_id, "/temp0");
	MountOrCreateDir("_TempData/" + title_id, "/temp");
}

static bool ClearDirectoryContents(const std::filesystem::path& dir) {
	bool ok = true;

	for (const auto& entry: Common::File::GetDirEntries(dir)) {
		if (entry.name == "." || entry.name == "..") {
			continue;
		}

		auto path = dir / entry.name;

		if (entry.is_file) {
			Common::File::RemoveReadonly(path);
			ok = Common::File::DeleteFile(path) && ok;
		} else {
			ok = ClearDirectoryContents(path) && ok;
			ok = Common::File::DeleteDirectory(path) && ok;
		}
	}

	return ok;
}

static void ClearDebugTextureFolder() {
	const std::string debug_texture_folder = "_Textures";

	if (!Common::File::IsDirectoryExisting(debug_texture_folder)) {
		Common::File::CreateDirectories(debug_texture_folder);
		return;
	}

	if (!ClearDirectoryContents(debug_texture_folder)) {
		LOGF_COLOR(Log::Color::BrightYellow, "TextureDump: failed to completely clear %s\n",
		           debug_texture_folder.c_str());
	}
}

static void Init(const Config::ConfigOptions& cfg, const std::filesystem::path& param_json,
                 Common::Subsystems& subsystems) {
	EXIT_IF(!Common::Thread::IsMainThread());

	subsystems.Initialize<Config::Lifecycle>();
	Config::Load(cfg);
	subsystems.Initialize<Log::Lifecycle>();

	if (Common::File::IsFileExisting(param_json)) {
		Loader::SystemContentLoadParamSfo(param_json);
		if (const auto flexible_memory_size = Loader::SystemContentGetFlexibleMemorySize();
		    flexible_memory_size != 0) {
			Libs::LibKernel::Memory::SetFlexibleMemorySize(flexible_memory_size);
		}
	}

	// Initialization order is explicit; destruction is automatic and reversed.
	subsystems.Initialize<Loader::Timer::Lifecycle>();
	subsystems.Initialize<Libs::LibKernel::PthreadLifecycle>();
	subsystems.Initialize<Profiler::Lifecycle>();
	subsystems.Initialize<Libs::Network::Lifecycle>();
	subsystems.Initialize<Libs::LibKernel::Memory::Lifecycle>();
	subsystems.Initialize<Libs::LibKernel::FileSystem::Lifecycle>();
	subsystems.Initialize<Libs::Controller::Lifecycle>();
	subsystems.Initialize<Libs::Audio::Lifecycle>();
	subsystems.Initialize<Libs::Graphics::Lifecycle>();
}

static void LoadElf(const std::filesystem::path& elf, bool dbg_print_reloc = false,
                    const std::filesystem::path& save_name = {}) {
	auto* rt = Common::Singleton<Loader::RuntimeLinker>::Instance();

	auto* program = rt->LoadProgram(
	    Libs::LibKernel::FileSystem::GetRealFilename(Common::PathToGenericString(elf)));

	if (dbg_print_reloc) {
		program->dbg_print_reloc = true;
	}

	if (!save_name.empty()) {
		rt->SaveProgram(program, Libs::LibKernel::FileSystem::GetRealFilename(
		                             Common::PathToGenericString(save_name)));
	}
}

// The folder a game's shader corpus lives in, and the folder the dump writes. Both are relative to
// the process's working directory for the same reason _ShaderCache is: each title gets its own.
static std::filesystem::path CorpusDirectory(const std::string& title_id) {
	return std::filesystem::path("_ShaderCorpus") / title_id;
}

static bool SetEnvironmentVariable(const char* name, const std::string& value) {
#if KYTY_PLATFORM == KYTY_PLATFORM_WINDOWS
	return ::_putenv_s(name, value.c_str()) == 0;
#else
	return ::setenv(name, value.c_str(), 1) == 0;
#endif
}

// --precompile: every shader the game's own files hold (Offline::EnumerateGame), brought to the
// point where a draw would hand it to the compiler, translated and stored in _ShaderCache by the
// draw's own path (PipelineCache::SeedOffline). The device is the one the game would have got:
// Libs::Graphics::Initialize ran in Init, before any of this.
//
// The shaders a game draws are a subset of the ones its files hold, and a draw whose own state
// differs from the offline stand-ins (ShaderOffline.h) compiles as it does today, so this is a
// head start and never a correctness question. The line per shader is machine readable: the
// launcher counts the DISTINCT line for its denominator and the PROGRESS lines for the bar.
static void PrecompileShaders(const std::filesystem::path& game, const std::string& title_id) {
	auto* cache = Libs::Graphics::WindowPipelineCache();
	EXIT_IF(cache == nullptr);

	std::string report;
	auto        shaders = Libs::Graphics::Offline::EnumerateGame(game, report);
	// The shaders the game actually handed to AGC on an earlier run, kept by ShaderDumpConsider
	// (libs/agc.cpp). They are the ones a game's own files do not hold, which for the engines that
	// keep them in their data files is most of the race or the level: without this, --precompile
	// covers the menu and the first minutes but nothing the game reaches later.
	auto corpus = std::filesystem::path {};
	if (const char* env = std::getenv("KYTY_SHADER_CORPUS"); env != nullptr && env[0] != '\0') {
		corpus = env;
	} else if (const auto kept = CorpusDirectory(title_id); std::filesystem::is_directory(kept)) {
		corpus = kept;
	}
	if (!corpus.empty()) {
		// The hash dedupe below drops the ones EnumerateGame already found.
		auto loaded = Libs::Graphics::Offline::LoadCorpus(corpus, report);
		shaders.insert(shaders.end(), std::make_move_iterator(loaded.begin()),
		               std::make_move_iterator(loaded.end()));
	}
	::printf("SHADERS\t%zu\n%s", shaders.size(), report.c_str());
	::fflush(stdout);

	// A container may carry a shader another one already carries: the emulator knows a shader by
	// XXH3 of its code alone, so the first copy of a hash stands for all of them. (Ceiling: two
	// containers holding the same code with *different* headers seed only the first, and a draw
	// using the other misses and compiles as it does today. Demon's Souls has 20954 distinct of
	// 47858.) The progress denominator is the distinct count, which is what has to be translated.
	std::unordered_set<uint64_t>                             seen;
	std::vector<const Libs::Graphics::Offline::ShaderBlob*>  distinct;
	distinct.reserve(shaders.size());
	for (const auto& blob: shaders) {
		if (seen.insert(Libs::Graphics::Offline::ShaderHash(blob)).second) {
			distinct.push_back(&blob);
		}
	}
	::printf("DISTINCT\t%zu\n", distinct.size());
	::fflush(stdout);

	// A count of skips says nothing about what to do about them: the reasons, counted, do. The
	// offline half can only stand in for the draws whose own state is its defaults (above), so the
	// list is the measure of what a corpus alone can never cover.
	std::map<std::string, size_t> skipped_why;

	size_t seeded  = 0;
	size_t skipped = 0;
	for (size_t i = 0; i < distinct.size(); i++) {
		Libs::Graphics::Offline::Program program;
		std::string                      note;
		std::string                      failure;
		const auto status = Libs::Graphics::Offline::PrepareOfflineProgram(*distinct[i], program, note,
		                                                                  failure);
		const bool ok = status == Libs::Graphics::Offline::Status::Ok && cache->SeedOffline(program);
		if (ok) {
			seeded++;
		} else {
			skipped++;
			skipped_why[failure.empty() ? "SeedOffline refused it" : failure]++;
		}
		::printf("PROGRESS\t%zu\t%zu\t%s\n", i + 1, distinct.size(), ok ? "seeded" : "skipped");
		::fflush(stdout);
	}
	cache->Save();
	for (const auto& [why, count]: skipped_why) {
		::printf("SKIP\t%zu\t%s\n", count, why.c_str());
	}
	::printf("PRECOMPILED\t%zu\t%zu\n", seeded, skipped);
	::fflush(stdout);

	// As the game path does after WindowRun: the cache is flushed and nothing is left to unwind in
	// order. (Letting the subsystems destruct here hangs with the device still up.)
	std::quick_exit(0);
}

static void Execute(const std::filesystem::path& game_patch) {
	auto           patch_path = game_patch;
	Common::Thread guest_thread(
	    [](void* param) {
		    auto* rt = Common::Singleton<Loader::RuntimeLinker>::Instance();
		    rt->Execute(*static_cast<const std::filesystem::path*>(param));
	    },
	    &patch_path);
	Libs::Graphics::WindowRun();
	std::quick_exit(0);
}

void Run(const RunOptions& options) {
	if (options.app0_dir.empty()) {
		EXIT("app0 directory is required\n");
	}

	if (options.elf.empty()) {
		EXIT("ELF is required\n");
	}

	const auto         param_json = options.app0_dir / "sce_sys" / "param.json";
	Common::Subsystems subsystems(true);
	if (options.precompile) {
		// The graphics subsystem creates the window during Init; this run never shows it.
		Libs::Graphics::WindowSetHidden();
	}
	Init(options.config, param_json, subsystems);

	ClearDebugTextureFolder();

	PrintSystemInfo();
	std::string title_id;
	if (Loader::SystemContentParamSfoGetString("TITLE_ID", &title_id) && !title_id.empty()) {
		Log::WriteToConsoleAndLog(fmt::format("Title ID: {}\n", title_id));
	}

	int ok = atexit(KytyClose);
	EXIT_NOT_IMPLEMENTED(ok != 0);

	if (options.precompile) {
		// The subsystems and the window unwind in the ordinary order as Run returns: nothing here
		// needs the quick exit the guest path takes.
		PrecompileShaders(options.app0_dir, title_id);
		return;
	}

	// Running a game also keeps what the game hands to AGC, in _ShaderCorpus/<title id>
	// (ShaderDumpConsider in libs/agc.cpp): a few thousand files of a few KB, one write per shader
	// the emulator has not seen, and the only way a later --precompile can cover what the game's own
	// files never hold. Set KYTY_SHADER_DUMP to another folder, or to 0 to turn the dump off.
	if (std::getenv("KYTY_SHADER_DUMP") == nullptr && !title_id.empty()) {
		SetEnvironmentVariable("KYTY_SHADER_DUMP",
		                       Common::PathToString(CorpusDirectory(title_id)));
	}

	// Guest threads are still running, so skip KytyClose() and only flush emergency state.
	ok = at_quick_exit(Common::Subsystems::EmergencyShutdownActive);
	EXIT_NOT_IMPLEMENTED(ok != 0);

	Libs::LibKernel::FileSystem::Mount(options.app0_dir, "/app0");
	Libs::LibKernel::FileSystem::Mount(options.app0_dir, "/hostapp");

	MountSandboxDirs();

	auto* rt = Common::Singleton<Loader::RuntimeLinker>::Instance();
	Libs::InitAll(rt->Symbols());

	LoadElf(options.elf);

	Execute(options.game_patch);
}

} // namespace Emulator
