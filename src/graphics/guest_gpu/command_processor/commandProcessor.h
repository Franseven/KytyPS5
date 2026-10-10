#ifndef GRAPHICS_GUEST_GPU_COMMAND_PROCESSOR_COMMAND_PROCESSOR_H
#define GRAPHICS_GUEST_GPU_COMMAND_PROCESSOR_COMMAND_PROCESSOR_H

#include "common/assert.h"
#include "common/platform/sysTimer.h"
#include "graphics/guest_gpu/hardwareContext.h"
#include "graphics/host_gpu/renderer/render.h"
#include "graphics/host_gpu/renderer/renderContext.h"

#include <cstdint>
#include <span>
#include <vector>

namespace Libs::Graphics {

bool TestWaitRegMemValue(uint64_t value, uint64_t ref, uint64_t mask, uint32_t func);

enum class Pm4ProcessResult { Complete, Blocked };

// Microsecond frame telemetry (QPC-based). GPU thread only: no atomics by design.
struct FrameTelemetry {
	uint64_t pm4_op_us[256] = {};
	uint64_t draw_record_us = 0;
	uint64_t draw_state_us = 0;
	// `draw_state_us` da solo dice quanto costa PrepareDrawRenderState, non *cosa* costa: la
	// chiave L1, la risoluzione dei programmi o la materializzazione dei render target. Sono
	// quattro strade diverse, quindi si contano separate.
	uint64_t draw_state_key_us = 0;
	uint64_t draw_state_programs_us = 0;
	// Dentro `draw_state_programs_us`: `GetGraphicsPrograms` non e' una cosa sola, e i tre pezzi
	// hanno fix diversi. `prepare` costruisce gli `input_info` dal draw (analisi del metadato AGC
	// dello shader, ogni volta, hit o miss). `parallel` e' la finestra in cui due worker
	// materializzano gli SRT mentre il draw aspetta. `lookup` e' la risoluzione vera -- se ha
	// dentro `MaterializeResources`, il costo e' il walk degli SRT, non la cache dei programmi.
	uint64_t draw_gp_prepare_vs_us = 0;
	uint64_t draw_gp_prepare_ps_us = 0;
	uint64_t draw_gp_parallel_us = 0;
	uint64_t draw_gp_lookup_vs_us = 0;
	uint64_t draw_gp_lookup_ps_us = 0;
	// Dentro `lookup`: il walk SRT (`MaterializeResources`) da solo. Il resto e' la chiave, la
	// ricerca nell'hash map, la scelta della permutazione.
	uint64_t draw_gp_walk_vs_us = 0;
	uint64_t draw_gp_walk_ps_us = 0;
	// Dentro `lookup`, fuori dal walk: la ricerca della permutazione (confronto della
	// `ResourceSpecialization` candidato per candidato) e la discesa nel percorso freddo (cache
	// disco, traduzione, compilazione). Il resto di `lookup` meno walk e meno perm e' la chiave, la
	// ricerca nell'hash map e la preparazione del runtime: roba da decine di nanosecondi.
	uint64_t draw_gp_perm_vs_us = 0;
	uint64_t draw_gp_perm_ps_us = 0;
	uint32_t draw_gp_perm_scanned = 0;
	uint32_t draw_gp_cold_calls = 0;
	// Dentro il walk: le fasi del refresh SRT (SrtPhaseTimings). Si accumulano solo se il walk gira
	// su *questo* thread: con KYTY_STAGE_PREP_PARALLEL=1 quello del vertex stage gira sull'helper e
	// le sue fasi non tornano qui (la somma delle fasi sara' minore del walk, ed e' voluto).
	uint64_t draw_gp_srt_setup_us = 0;
	uint64_t draw_gp_srt_flat_us = 0;
	uint64_t draw_gp_srt_buffers_us = 0;
	uint64_t draw_gp_srt_images_us = 0;
	uint64_t draw_gp_srt_rest_us = 0;
	// Il percorso preso dal walk: replay dei soli walk_slots (veloce) contro walk completo dei
	// blocchi. Se `replay` e' vicino al numero di walk, il costo dei flat slot e' il walker; se
	// domina `key_miss` il problema e' la chiave del replay, non il walker.
	uint32_t draw_gp_walk_replay = 0;
	uint32_t draw_gp_walk_key_miss = 0;
	uint32_t draw_gp_walk_outcome_miss = 0;
	uint32_t draw_gp_walk_full = 0;
	uint32_t draw_gp_walk_empty = 0;
	uint32_t draw_gp_walk_flat_slots = 0;
	uint64_t draw_state_resolve_us = 0;
	uint32_t draw_state_hits = 0;
	uint32_t draw_state_misses = 0;
	// Sonda di correttezza per il salto di `RefreshShaders`: fra i draw la cui `pre_key` combacia
	// con la voce in cache, quanti hanno ritrovato *lo stesso* stato ricalcolato, e quanti no.
	// Misurato in una scena da 1376 draw/frame: 724 diff su 89.952 hit pieni = **0,805%**, quindi
	// saltare `RefreshShaders` alla cieca e' sbagliato su un draw su 124. Serve sapere *quale* campo
	// differisce per sapere cosa aggiungere alla chiave: tre contatori, non uno.
	uint32_t draw_state_verify_hits = 0;
	uint32_t draw_state_verify_diff = 0;
	uint32_t draw_state_verify_diff_vertex = 0;
	uint32_t draw_state_verify_diff_pixel = 0;
	uint32_t draw_state_verify_diff_programs = 0;
	// I due modi in cui un programma puo' cambiare, e sono cause *diverse*:
	//   id   -- `ShaderProgram.id` e' l'handle della permutazione. Cambia quando la
	//           `ResourceSpecialization` del draw non combacia piu' con quella della sorgente in
	//           cache, cioe' quando cambia la *memoria guest* che lo shader legge. Non c'e' modo
	//           di prevederlo da un registro: se sono tutti id, saltare `RefreshShaders` non e'
	//           solo sbagliato, e' non recuperabile con una chiave.
	//   hash -- `options.shader_hash` per il VS non e' solo l'hash di `es_regs.data_addr`: sul
	//           percorso GS-front e' XXH3 di {front, back}, dove back sta in `gs_regs.data_addr`.
	//           Quel campo NON e' nella `pre_key`. Se i diff sono hash, la chiave si completa
	//           aggiungendo gli indirizzi degli stadi che la composizione legge.
	uint32_t draw_state_verify_diff_programs_id = 0;
	uint32_t draw_state_verify_diff_programs_hash = 0;
	// Di quei disallineamenti, quanti sono solo il layout dei vertici -- cioe' esattamente cio' che
	// il post-check gia' intercetta. Se `diff == layout_diff` la sola cosa che la `pre_key` non
	// copre e' il layout, e allora basta calcolare quello per salvare i programmi.
	uint32_t draw_state_verify_layout_diff = 0;
	uint64_t draw_bindings_us = 0;
	// Dentro `draw_bindings_us`, che e' il 23% del frame su GT7 e non era mai stato aperto:
	// la preparazione dei descrittori (risoluzione delle immagini dei descrittori, dei target di
	// render, dei buffer) e il commit nella command buffer Vulkan sono due costi con fix opposti --
	// il primo si memoizza, il secondo si registra piu' tardi o si fonde.
	uint64_t draw_bind_stage_prep_us = 0; // PrepareBindings, una volta per stadio
	uint64_t draw_bind_find_us = 0;       // FindBuffers: risoluzione degli indirizzi guest
	uint64_t draw_bind_images_us = 0;     // RebindImages: le immagini dei descrittori
	uint64_t draw_bind_targets_us = 0;    // i color target: FindImage + BindRenderTarget
	uint64_t draw_bind_rebind_us = 0;     // RebindBuffers, l'ultimo giro
	// Dentro `draw_bind_rebind_us`: il giro costruisce i buffer e poi *carica* in un ring host
	// visibile la tabella SRT appiattita e i dati di shader, una volta per stadio per draw. Sapere
	// quanto di quel millisecondo e' la copia e' la differenza fra un memo sul contenuto (che
	// richiede un confronto) e niente affatto.
	uint64_t draw_bind_sources_us = 0;  // NativeStorageBuffer + word del descrittore, per buffer
	uint64_t draw_bind_upsrt_us = 0;    // NativeUpload + FillFlatSlotsOnGpu della SRT appiattita
	uint64_t draw_bind_updata_us = 0;   // NativeUpload dei dati di shader
	uint64_t draw_bind_commit_us = 0;     // CommitBindings: le chiamate vkCmd*
	uint64_t draw_vertex_us = 0;
	uint64_t draw_pipe_us = 0;
	uint64_t draw_emit_us = 0;
	uint64_t wait_readmem_us = 0;
	uint64_t wait_flip_us = 0;
	uint64_t wait_gc_us = 0;
	uint32_t draws = 0;
	uint32_t flushes = 0;
};

inline FrameTelemetry& GetFrameTelemetry() {
	static thread_local FrameTelemetry instance;
	return instance;
}

inline uint64_t TelemetryNowUs() {
	uint64_t counter = 0;
	uint64_t frequency = 0;
	SysQueryPerformanceCounter(&counter);
	SysQueryPerformanceFrequency(&frequency);
	if (frequency == 0) {
		return 0;
	}
	return counter * 1000000ull / frequency;
}

inline void TelemetryAddPm4Op(uint8_t opcode, uint64_t delta_us) {
	GetFrameTelemetry().pm4_op_us[opcode] += delta_us;
}

inline void TelemetryAddDrawRecord(uint64_t delta_us) {
	GetFrameTelemetry().draw_record_us += delta_us;
}

inline void TelemetryAddDrawState(uint64_t delta_us) {
	GetFrameTelemetry().draw_state_us += delta_us;
}

inline void TelemetryAddDrawStateKey(uint64_t delta_us) {
	GetFrameTelemetry().draw_state_key_us += delta_us;
}

inline void TelemetryAddDrawStatePrograms(uint64_t delta_us) {
	GetFrameTelemetry().draw_state_programs_us += delta_us;
}

inline void TelemetryAddDrawStateResolve(uint64_t delta_us) {
	GetFrameTelemetry().draw_state_resolve_us += delta_us;
}

inline void TelemetryAddGpPrepareVs(uint64_t delta_us) {
	GetFrameTelemetry().draw_gp_prepare_vs_us += delta_us;
}

inline void TelemetryAddGpPreparePs(uint64_t delta_us) {
	GetFrameTelemetry().draw_gp_prepare_ps_us += delta_us;
}

inline void TelemetryAddGpParallel(uint64_t delta_us) {
	GetFrameTelemetry().draw_gp_parallel_us += delta_us;
}

inline void TelemetryAddGpLookupVs(uint64_t delta_us) {
	GetFrameTelemetry().draw_gp_lookup_vs_us += delta_us;
}

inline void TelemetryAddGpLookupPs(uint64_t delta_us) {
	GetFrameTelemetry().draw_gp_lookup_ps_us += delta_us;
}

inline void TelemetryAddGpWalkVs(uint64_t delta_us) {
	GetFrameTelemetry().draw_gp_walk_vs_us += delta_us;
}

inline void TelemetryAddGpWalkPs(uint64_t delta_us) {
	GetFrameTelemetry().draw_gp_walk_ps_us += delta_us;
}

inline void TelemetryAddGpPermVs(uint64_t delta_us) {
	GetFrameTelemetry().draw_gp_perm_vs_us += delta_us;
}

inline void TelemetryAddGpPermPs(uint64_t delta_us) {
	GetFrameTelemetry().draw_gp_perm_ps_us += delta_us;
}

inline void TelemetryAddGpPermScanned(uint32_t candidates) {
	GetFrameTelemetry().draw_gp_perm_scanned += candidates;
}

inline void TelemetryCountGpCold() {
	GetFrameTelemetry().draw_gp_cold_calls++;
}

inline void TelemetryCountDrawStateHit() {
	GetFrameTelemetry().draw_state_hits++;
}

inline void TelemetryCountDrawStateMiss() {
	GetFrameTelemetry().draw_state_misses++;
}

inline void TelemetryCountDrawStateVerify(uint32_t diff_mask, bool layout_diff) {
	auto& telemetry = GetFrameTelemetry();
	telemetry.draw_state_verify_hits++;
	if (diff_mask != 0) {
		telemetry.draw_state_verify_diff++;
	}
	if ((diff_mask & 1u) != 0) {
		telemetry.draw_state_verify_diff_vertex++;
	}
	if ((diff_mask & 2u) != 0) {
		telemetry.draw_state_verify_diff_pixel++;
	}
	if ((diff_mask & 4u) != 0) {
		telemetry.draw_state_verify_diff_programs++;
	}
	if ((diff_mask & 8u) != 0) {
		telemetry.draw_state_verify_diff_programs_id++;
	}
	if ((diff_mask & 16u) != 0) {
		telemetry.draw_state_verify_diff_programs_hash++;
	}
	if (layout_diff) {
		telemetry.draw_state_verify_layout_diff++;
	}
}

inline void TelemetryAddDrawBindings(uint64_t delta_us) {
	GetFrameTelemetry().draw_bindings_us += delta_us;
}

inline void TelemetryAddDrawVertex(uint64_t delta_us) {
	GetFrameTelemetry().draw_vertex_us += delta_us;
}

inline void TelemetryAddDrawPipe(uint64_t delta_us) {
	GetFrameTelemetry().draw_pipe_us += delta_us;
}

inline void TelemetryAddBindStagePrep(uint64_t delta_us) {
	GetFrameTelemetry().draw_bind_stage_prep_us += delta_us;
}

inline void TelemetryAddBindFind(uint64_t delta_us) {
	GetFrameTelemetry().draw_bind_find_us += delta_us;
}

inline void TelemetryAddBindImages(uint64_t delta_us) {
	GetFrameTelemetry().draw_bind_images_us += delta_us;
}

inline void TelemetryAddBindTargets(uint64_t delta_us) {
	GetFrameTelemetry().draw_bind_targets_us += delta_us;
}

inline void TelemetryAddBindRebind(uint64_t delta_us) {
	GetFrameTelemetry().draw_bind_rebind_us += delta_us;
}

inline void TelemetryAddBindSources(uint64_t delta_us) {
	GetFrameTelemetry().draw_bind_sources_us += delta_us;
}

inline void TelemetryAddBindUpSrt(uint64_t delta_us) {
	GetFrameTelemetry().draw_bind_upsrt_us += delta_us;
}

inline void TelemetryAddBindUpData(uint64_t delta_us) {
	GetFrameTelemetry().draw_bind_updata_us += delta_us;
}

inline void TelemetryAddBindCommit(uint64_t delta_us) {
	GetFrameTelemetry().draw_bind_commit_us += delta_us;
}

inline void TelemetryAddDrawEmit(uint64_t delta_us) {
	GetFrameTelemetry().draw_emit_us += delta_us;
}

inline void TelemetryAddWaitReadMem(uint64_t delta_us) {
	GetFrameTelemetry().wait_readmem_us += delta_us;
}

inline void TelemetryAddWaitFlip(uint64_t delta_us) {
	GetFrameTelemetry().wait_flip_us += delta_us;
}

inline void TelemetryAddWaitGC(uint64_t delta_us) {
	GetFrameTelemetry().wait_gc_us += delta_us;
}

inline void TelemetryCountDraw() {
	GetFrameTelemetry().draws++;
}

inline void TelemetryCountFlush() {
	GetFrameTelemetry().flushes++;
}

void TelemetryEndFrame();

enum class ContextStateOperation : uint32_t {
	Clear     = 0,
	Push      = 1,
	Pop       = 2,
	PushClear = 3,
};

class Pm4Execution {
public:
	[[nodiscard]] bool MadeProgress() const noexcept { return m_made_progress; }

private:
	friend class CommandProcessor;

	struct BufferCursor {
		std::span<const uint32_t> commands;
		uint32_t                  offset_dw = 0;
	};

	std::vector<BufferCursor> m_buffer_stack;
	std::span<const uint32_t> m_next_buffer;
	bool                      m_chain         = false;
	bool                      m_suspended     = false;
	bool                      m_made_progress = false;
};

class CommandProcessor {
public:
	struct FlipInfo {
		int     handle    = 0;
		int     index     = 0;
		int     flip_mode = 0;
		int64_t flip_arg  = 0;
	};

	CommandProcessor(RenderContext& renderer, int interrupt_event_id)
	    : m_renderer(renderer), m_interrupt_event_id(interrupt_event_id) {}
	~CommandProcessor() = default;

	KYTY_CLASS_NO_COPY(CommandProcessor);

	void Reset();
	void ApplyContextStateOperation(ContextStateOperation operation);

	void            BufferInit();
	void            BufferFlush();
	void            BufferWait();
	// See CommandScheduler::CompleteReleaseMemWrite -- only safe for a RELEASE_MEM that already
	// wrote its guest-visible value synchronously and scheduled no interrupt callback.
	void            CompleteReleaseMemWrite();
	// See CommandScheduler::CompleteReleaseMemInterrupt -- for a RELEASE_MEM that DOES request a
	// guest interrupt/event; uses a smaller batch bound as a hedge against delaying it.
	void            CompleteReleaseMemInterrupt();
	// See CommandScheduler::CompleteDraw -- called after every DrawIndex/DrawAuto.
	void            CompleteDraw();
	HW::Context&    GetCtx() { return m_ctx; }
	HW::UserConfig& GetUcfg() { return m_ucfg; }
	HW::Shader&     GetShCtx() { return m_sh_ctx; }

	void SetIndexType(uint32_t index_type_and_size);
	void SetIndexBaseAddress(uint64_t index_base_addr);
	void SetIndexBufferSize(uint32_t index_buffer_size);
	void SetDrawIndirectArgsBaseAddress(uint64_t draw_indirect_args_base_addr);
	void SetDispatchIndirectArgsBaseAddress(uint64_t dispatch_indirect_args_base_addr);
	[[nodiscard]] uint64_t GetDispatchIndirectArgsBaseAddress() const {
		return m_dispatch_indirect_args_base_addr;
	}
	void SetNumInstances(uint32_t num_instances);
	void DrawIndex(DrawIndexArgs args);
	void DrawIndexOffset(uint32_t index_offset, uint32_t index_count);
	void DrawIndexAuto(DrawAutoArgs args);
	void DrawIndirect(uint32_t data_offset, uint32_t draw_initiator, bool indexed);
	void DrawIndirectMulti(uint32_t data_offset, uint32_t max_count_or_count,
	                       const volatile uint32_t* count_addr, uint32_t stride_in_bytes,
	                       uint32_t draw_initiator, bool indexed, uint32_t draw_index_register);
	void WriteAtEndOfPipe32(uint32_t cache_policy, uint32_t event_write_dest,
	                        uint32_t eop_event_type, uint32_t cache_action, uint32_t event_index,
	                        uint32_t event_write_source, void* dst_gpu_addr, uint32_t value,
	                        uint32_t interrupt_selector, uint32_t interrupt_context_id = 0);
	void WriteAtEndOfPipe64(uint32_t cache_policy, uint32_t event_write_dest,
	                        uint32_t eop_event_type, uint32_t cache_action, uint32_t event_index,
	                        uint32_t event_write_source, void* dst_gpu_addr, uint64_t value,
	                        uint32_t interrupt_selector, uint32_t interrupt_context_id = 0);
	void Flip();
	void Flip(void* dst_gpu_addr, uint32_t value);
	void FlipWithInterrupt(uint32_t eop_event_type, uint32_t cache_action, void* dst_gpu_addr,
	                       uint32_t value);
	void PrepareCpuFlip(uint64_t request_id);
	void SynchronizeGpu();
	void EmitGlobalBarrier();
	void TriggerEopEventAtEndOfPipe(uint32_t interrupt_context_id);
	void DispatchDirect(uint32_t thread_group_x, uint32_t thread_group_y, uint32_t thread_group_z,
	                    uint32_t mode);
	void DispatchIndirect(uint64_t args_addr, uint32_t mode);
	void WaitFlipDone(uint32_t video_out_handle, uint32_t display_buffer_index);
	void TriggerEvent(uint32_t event_type, uint32_t event_index, uint64_t event_address = 0);

	void SetUserDataMarker(HW::UserSgprType type) { m_user_data_marker = type; }
	[[nodiscard]] HW::UserSgprType GetUserDataMarker() const { return m_user_data_marker; }

	void ResetDeCe();
	void SetCeComplete(bool complete) { m_ce_complete = complete; }
	void WaitCe();
	void WaitDeDiff(uint32_t diff);
	void WaitForRewind(bool valid);
	void IncrementDe();
	void IncrementCe();

	void WriteConstRam(uint32_t offset, const uint32_t* src, uint32_t dw_num);
	void DumpConstRam(uint32_t* dst, uint32_t offset, uint32_t dw_num);

	template <typename T>
	void WaitRegMem(uint32_t func, const T* addr, T ref, T mask, uint32_t poll, uint32_t wait_op);
	// A label as the command processor sees it: end-of-pipe writes count once recorded, even
	// before the CPU can see them (GuestGpu::LabelsAtCompletion).
	template <typename T>
	[[nodiscard]] T ReadLabel(const volatile T* addr) const;
	void WriteData(uint32_t* dst, const uint32_t* src, uint32_t dw_num, uint32_t write_control);
	void WriteReferenceClock(uint64_t dst_address, uint32_t num_bytes);
	void DmaData(uint8_t engine, uint8_t dst_sel, uint8_t dst_cache_policy,
	             uint64_t dst_address_or_offset, uint8_t src_sel, uint8_t src_cache_policy,
	             uint64_t src_address_or_offset_or_immediate, uint32_t num_bytes,
	             uint8_t wait_for_previous, uint8_t write_confirm, uint8_t block_engine);
	void SetPredication(uint32_t condition, uint32_t op, uint32_t wait_op,
	                    const volatile void* address, uint32_t count_in_dwords);
	[[nodiscard]] bool ShouldSkipPredicatedPackets() const { return m_predicate_skip; }

	Pm4ProcessResult Process(Pm4Execution& execution, std::span<const uint32_t> commands);
	void             ProcessIndirectBuffer(std::span<const uint32_t> commands, bool chain);

	void SetFlip(const FlipInfo& flip) { m_flip = flip; }

	[[nodiscard]] uint64_t GetSubmitId() const { return m_submit_id; }
	void                   SetSubmitId(uint64_t submit_id) { m_submit_id = submit_id; }
	[[nodiscard]] bool     IsAsyncComputeQueue() const { return m_interrupt_event_id >= 0x20; }

private:
	template <typename T>
	void WriteAtEndOfPipe(uint32_t cache_policy, uint32_t event_write_dest, uint32_t eop_event_type,
	                      uint32_t cache_action, uint32_t event_index, uint32_t event_write_source,
	                      void* dst_gpu_addr, T value, uint32_t interrupt_selector,
	                      uint32_t interrupt_context_id);
	void ProcessPm4(Pm4Execution& execution);
	void SuspendPm4();
	CommandScheduler&   GetScheduler() const { return m_renderer.GetCommandScheduler(); }
	CommandBuffer&      CurrentBuffer() { return GetScheduler().Current(); }

	RenderContext&   m_renderer;
	HW::Context      m_ctx;
	HW::Context      m_saved_ctx;
	bool             m_context_state_pushed = false;
	HW::UserConfig   m_ucfg;
	HW::Shader       m_sh_ctx;
	HW::UserSgprType m_user_data_marker                 = HW::UserSgprType::Unknown;
	uint32_t         m_index_type_and_size              = 0;
	uint32_t         m_index_buffer_size                = 0;
	uint64_t         m_index_base_addr                  = 0;
	uint64_t         m_draw_indirect_args_base_addr     = 0;
	uint64_t         m_dispatch_indirect_args_base_addr = 0;
	// Persistent draw state: indirect draws update it for subsequent draws.
	uint32_t m_num_instances = 1;
	// The DrawIndirectArgs whose instance count NUM_INSTANCES holds after a GPU-read indirect draw;
	// read on the CPU only if a later draw takes its instance count from there.
	uint64_t m_num_instances_args = 0;
	[[nodiscard]] uint32_t NumInstances();

	uint32_t m_de_count    = 0;
	uint32_t m_ce_count    = 0;
	bool     m_ce_complete = false;

	uint32_t m_const_ram[0x3000] = {0};

	FlipInfo  m_flip;
	const int m_interrupt_event_id;
	uint64_t  m_submit_id                   = 0;
	uint64_t  m_synthetic_occlusion_counter = 0;
	bool      m_predicate_skip              = false;
};

} // namespace Libs::Graphics

#endif // GRAPHICS_GUEST_GPU_COMMAND_PROCESSOR_COMMAND_PROCESSOR_H
