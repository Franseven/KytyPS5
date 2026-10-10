// The offline half of the shader path: see ShaderOffline.h. This is the code tests/ShaderBatchTool
// .cpp grew, moved here so the emulator's own precompiler uses the same extraction and the same
// stand-in state as the batch tool: one translation, one set of cache keys.

#include "graphics/shader/ShaderOffline.h"

#include "common/logging/log.h"
#include "graphics/guest_gpu/gpu_defs.h"
#include "graphics/guest_gpu/pm4.h"
#include "graphics/shader/recompiler/ShaderRecompiler.h"
#include "graphics/shader/shaderVertexMetadata.h"
#include "libs/agc.h"

#include <algorithm>
#include <cstring>
#include <fmt/format.h>
#include <fstream>
#include <iterator>
#include <malloc.h>
#include <map>
#include <stdexcept>
#include <xxhash.h>

namespace Libs::Graphics::Offline {
// Shader checksum registers (COMPUTE_SHADER_CHKSUM, SPI_SHADER_PGM_CHKSUM_PS/GS): no state.
constexpr uint32_t ComputeChecksum = 0x22a;
constexpr uint32_t PixelChecksum   = 0x006;
constexpr uint32_t GeometryChecksum = 0x080;


// Kept alive for the whole run: the shader map and the hash cache hold these addresses.
void* CopyAligned(const std::vector<char>& bytes, size_t alignment) {
	void* memory = _aligned_malloc(std::max<size_t>(bytes.size(), 1), alignment);
	std::memcpy(memory, bytes.data(), bytes.size());
	return memory;
}

struct GuestState {
	HW::Context           context;
	HW::UserConfig        user_config;
	HW::ComputeShaderInfo compute;
	HW::PixelShaderInfo   pixel;
	HW::VertexShaderInfo  vertex;
	uint32_t              dispatch_modifier = 0;
	std::string           unhandled;
};

void NoteUnhandled(GuestState& state, const char* space, uint32_t offset) {
	state.unhandled += fmt::format("{}{}:0x{:03x}", state.unhandled.empty() ? "" : " ", space, offset);
}

uint64_t SetAddressLo(uint64_t address, uint32_t value) {
	return (address & 0xFFFFFF00000000FFull) | (static_cast<uint64_t>(value) << 8u);
}

uint64_t SetAddressHi(uint64_t address, uint32_t value) {
	return (address & 0xFFFF00FFFFFFFFFFull) | ((static_cast<uint64_t>(value) & 0xffu) << 40u);
}

// The context registers the headers carry, decoded as pm4Handlers.cpp does.
void SetContextRegister(GuestState& state, uint32_t offset, uint32_t value) {
	auto& ctx = state.context;
	switch (offset) {
		case Pm4::SPI_PS_INPUT_ENA: ctx.SetPsInputEna(value); break;
		case Pm4::SPI_PS_INPUT_ADDR: ctx.SetPsInputAddr(value); break;
		case Pm4::SPI_PS_IN_CONTROL: ctx.SetPsInControl(value); break;
		case Pm4::SPI_BARYC_CNTL: ctx.SetBarycCntl(value); break;
		case Pm4::SPI_SHADER_Z_FORMAT: ctx.SetShaderZFormat(value); break;
		case Pm4::SPI_SHADER_COL_FORMAT:
			for (uint32_t i = 0; i < 8; i++) {
				ctx.SetTargetOutputMode(i, (value >> (i * 4)) & 0xFu);
			}
			break;
		case Pm4::DB_SHADER_CONTROL: {
			HW::DepthShaderControl control {};
			control.other_bits = value & 0xFFFF908Eu;
			control.conservative_z_export_value =
			    KYTY_PM4_GET(value, DB_SHADER_CONTROL, CONSERVATIVE_Z_EXPORT);
			control.shader_z_behavior  = KYTY_PM4_GET(value, DB_SHADER_CONTROL, Z_ORDER);
			control.shader_kill_enable = KYTY_PM4_GET(value, DB_SHADER_CONTROL, KILL_ENABLE) != 0;
			control.shader_z_export_enable =
			    KYTY_PM4_GET(value, DB_SHADER_CONTROL, Z_EXPORT_ENABLE) != 0;
			control.shader_mask_export_enable =
			    KYTY_PM4_GET(value, DB_SHADER_CONTROL, MASK_EXPORT_ENABLE) != 0;
			control.shader_dual_export_enable =
			    KYTY_PM4_GET(value, DB_SHADER_CONTROL, DUAL_EXPORT_ENABLE) != 0;
			control.shader_execute_on_noop =
			    KYTY_PM4_GET(value, DB_SHADER_CONTROL, EXEC_ON_NOOP) != 0;
			control.alpha_to_mask_disable =
			    KYTY_PM4_GET(value, DB_SHADER_CONTROL, ALPHA_TO_MASK_DISABLE) != 0;
			ctx.SetDepthShaderControl(control);
			break;
		}
		case Pm4::CB_SHADER_MASK: ctx.SetShaderMask(value); break;
		case Pm4::PA_SC_SHADER_CONTROL: ctx.SetScShaderControl(value); break;
		case Pm4::SPI_VS_OUT_CONFIG: ctx.SetVsOutConfig(value); break;
		case Pm4::SPI_SHADER_POS_FORMAT: ctx.SetShaderPosFormat(value); break;
		case Pm4::SPI_SHADER_IDX_FORMAT: ctx.SetShaderIdxFormat(value); break;
		case Pm4::PA_CL_VS_OUT_CNTL: ctx.SetClVsOutCntl(value); break;
		case Pm4::GE_NGG_SUBGRP_CNTL: ctx.SetNggSubgrpCntl(value); break;
		case Pm4::VGT_GS_INSTANCE_CNT: ctx.SetGsInstanceCnt(value); break;
		case Pm4::VGT_GS_ONCHIP_CNTL: ctx.SetGsOnchipCntl(value); break;
		case Pm4::GE_MAX_OUTPUT_PER_SUBGROUP: ctx.SetMaxOutputPerSubgroup(value); break;
		case Pm4::VGT_ESGS_RING_ITEMSIZE: ctx.SetEsgsRingItemsize(value); break;
		case Pm4::VGT_GS_MAX_VERT_OUT: ctx.SetGsMaxVertOut(value); break;
		case Pm4::VGT_SHADER_STAGES_EN: ctx.SetShaderStages(value); break;
		case Pm4::VGT_GS_OUT_PRIM_TYPE: ctx.SetGsOutPrimType(value); break;
		default: NoteUnhandled(state, "cx", offset); break;
	}
}

// The SH registers the headers carry, decoded as pm4Handlers.cpp does.
void SetShRegister(GuestState& state, uint32_t offset, uint32_t value) {
	auto& cs = state.compute.cs_regs;
	auto& ps = state.pixel.ps_regs;
	auto& vs = state.vertex;
	switch (offset) {
		case Pm4::COMPUTE_PGM_LO: cs.data_addr = SetAddressLo(cs.data_addr, value); break;
		case Pm4::COMPUTE_PGM_HI: cs.data_addr = SetAddressHi(cs.data_addr, value); break;
		case Pm4::COMPUTE_PGM_RSRC1:
			cs.vgprs =
			    (value >> Pm4::COMPUTE_PGM_RSRC1_VGPRS_SHIFT) & Pm4::COMPUTE_PGM_RSRC1_VGPRS_MASK;
			cs.priority = (value >> Pm4::COMPUTE_PGM_RSRC1_PRIORITY_SHIFT) &
			              Pm4::COMPUTE_PGM_RSRC1_PRIORITY_MASK;
			cs.float_mode = (value >> Pm4::COMPUTE_PGM_RSRC1_FLOAT_MODE_SHIFT) &
			                Pm4::COMPUTE_PGM_RSRC1_FLOAT_MODE_MASK;
			cs.dx10_clamp = ((value >> Pm4::COMPUTE_PGM_RSRC1_DX10_CLAMP_SHIFT) &
			                 Pm4::COMPUTE_PGM_RSRC1_DX10_CLAMP_MASK) != 0u;
			cs.debug_mode = ((value >> Pm4::COMPUTE_PGM_RSRC1_DEBUG_MODE_SHIFT) &
			                 Pm4::COMPUTE_PGM_RSRC1_DEBUG_MODE_MASK) != 0u;
			cs.ieee_mode = ((value >> Pm4::COMPUTE_PGM_RSRC1_IEEE_MODE_SHIFT) &
			                Pm4::COMPUTE_PGM_RSRC1_IEEE_MODE_MASK) != 0u;
			cs.fp16_overflow = ((value >> Pm4::COMPUTE_PGM_RSRC1_FP16_OVFL_SHIFT) &
			                    Pm4::COMPUTE_PGM_RSRC1_FP16_OVFL_MASK) != 0u;
			cs.threadgroup_configuration = ((value >> Pm4::COMPUTE_PGM_RSRC1_WGP_MODE_SHIFT) &
			                                Pm4::COMPUTE_PGM_RSRC1_WGP_MODE_MASK) != 0u;
			cs.require_forward_progress = ((value >> Pm4::COMPUTE_PGM_RSRC1_FWD_PROGRESS_SHIFT) &
			                               Pm4::COMPUTE_PGM_RSRC1_FWD_PROGRESS_MASK) != 0u;
			break;
		case Pm4::COMPUTE_PGM_RSRC2:
			cs.scratch_en = ((value >> Pm4::COMPUTE_PGM_RSRC2_SCRATCH_EN_SHIFT) &
			                 Pm4::COMPUTE_PGM_RSRC2_SCRATCH_EN_MASK) != 0u;
			cs.user_sgpr = (value >> Pm4::COMPUTE_PGM_RSRC2_USER_SGPR_SHIFT) &
			               Pm4::COMPUTE_PGM_RSRC2_USER_SGPR_MASK;
			cs.tgid_x_en = ((value >> Pm4::COMPUTE_PGM_RSRC2_TGID_X_EN_SHIFT) &
			                Pm4::COMPUTE_PGM_RSRC2_TGID_X_EN_MASK) != 0u;
			cs.tgid_y_en = ((value >> Pm4::COMPUTE_PGM_RSRC2_TGID_Y_EN_SHIFT) &
			                Pm4::COMPUTE_PGM_RSRC2_TGID_Y_EN_MASK) != 0u;
			cs.tgid_z_en = ((value >> Pm4::COMPUTE_PGM_RSRC2_TGID_Z_EN_SHIFT) &
			                Pm4::COMPUTE_PGM_RSRC2_TGID_Z_EN_MASK) != 0u;
			cs.tg_size_en = ((value >> Pm4::COMPUTE_PGM_RSRC2_TG_SIZE_EN_SHIFT) &
			                 Pm4::COMPUTE_PGM_RSRC2_TG_SIZE_EN_MASK) != 0u;
			cs.tidig_comp_cnt = (value >> Pm4::COMPUTE_PGM_RSRC2_TIDIG_COMP_CNT_SHIFT) &
			                    Pm4::COMPUTE_PGM_RSRC2_TIDIG_COMP_CNT_MASK;
			cs.lds_size =
			    (value >> Pm4::COMPUTE_PGM_RSRC2_LDS_SIZE_SHIFT) & Pm4::COMPUTE_PGM_RSRC2_LDS_SIZE_MASK;
			break;
		case Pm4::COMPUTE_PGM_RSRC3:
			cs.shared_vgprs = (value >> Pm4::COMPUTE_PGM_RSRC3_SHARED_VGPRS_SHIFT) &
			                  Pm4::COMPUTE_PGM_RSRC3_SHARED_VGPRS_MASK;
			break;
		case Pm4::COMPUTE_NUM_THREAD_X: cs.num_thread_x = value; break;
		case Pm4::COMPUTE_NUM_THREAD_Y: cs.num_thread_y = value; break;
		case Pm4::COMPUTE_NUM_THREAD_Z: cs.num_thread_z = value; break;
		case Pm4::SPI_SHADER_PGM_LO_PS: ps.data_addr = SetAddressLo(ps.data_addr, value); break;
		case Pm4::SPI_SHADER_PGM_HI_PS: ps.data_addr = SetAddressHi(ps.data_addr, value); break;
		case Pm4::SPI_SHADER_PGM_RSRC1_PS: {
			HW::PsShaderResource1 r1;
			r1.vgprs            = KYTY_PM4_GET(value, SPI_SHADER_PGM_RSRC1_PS, VGPRS);
			r1.priority         = KYTY_PM4_GET(value, SPI_SHADER_PGM_RSRC1_PS, PRIORITY);
			r1.float_mode       = KYTY_PM4_GET(value, SPI_SHADER_PGM_RSRC1_PS, FLOAT_MODE);
			r1.dx10_clamp       = KYTY_PM4_GET(value, SPI_SHADER_PGM_RSRC1_PS, DX10_CLAMP) != 0;
			r1.debug_mode       = KYTY_PM4_GET(value, SPI_SHADER_PGM_RSRC1_PS, DEBUG_MODE) != 0;
			r1.ieee_mode        = KYTY_PM4_GET(value, SPI_SHADER_PGM_RSRC1_PS, IEEE_MODE) != 0;
			r1.cu_group_disable = KYTY_PM4_GET(value, SPI_SHADER_PGM_RSRC1_PS, CU_GROUP_DISABLE) != 0;
			r1.require_forward_progress =
			    KYTY_PM4_GET(value, SPI_SHADER_PGM_RSRC1_PS, FWD_PROGRESS) != 0;
			r1.fp16_overflow = KYTY_PM4_GET(value, SPI_SHADER_PGM_RSRC1_PS, FP16_OVFL) != 0;
			ps.rsrc1         = r1;
			break;
		}
		case Pm4::SPI_SHADER_PGM_RSRC2_PS: {
			HW::PsShaderResource2 r2;
			r2.scratch_en = KYTY_PM4_GET(value, SPI_SHADER_PGM_RSRC2_PS, SCRATCH_EN);
			r2.user_sgpr  = KYTY_PM4_GET(value, SPI_SHADER_PGM_RSRC2_PS, USER_SGPR) +
			                (KYTY_PM4_GET(value, SPI_SHADER_PGM_RSRC2_PS, USER_SGPR_MSB) << 5u);
			r2.wave_cnt_en    = KYTY_PM4_GET(value, SPI_SHADER_PGM_RSRC2_PS, WAVE_CNT_EN);
			r2.extra_lds_size = KYTY_PM4_GET(value, SPI_SHADER_PGM_RSRC2_PS, EXTRA_LDS_SIZE);
			r2.raster_ordered_shading =
			    KYTY_PM4_GET(value, SPI_SHADER_PGM_RSRC2_PS, RASTER_ORDERED_SHADING);
			r2.shared_vgprs = KYTY_PM4_GET(value, SPI_SHADER_PGM_RSRC2_PS, SHARED_VGPR_CNT);
			ps.rsrc2        = r2;
			break;
		}
		case Pm4::SPI_SHADER_PGM_LO_ES:
			vs.es_regs.data_addr = SetAddressLo(vs.es_regs.data_addr, value);
			break;
		case Pm4::SPI_SHADER_PGM_HI_ES:
			vs.es_regs.data_addr = SetAddressHi(vs.es_regs.data_addr, value);
			break;
		case Pm4::SPI_SHADER_PGM_RSRC1_GS: {
			HW::GsShaderResource1 r1;
			r1.vgprs           = KYTY_PM4_GET(value, SPI_SHADER_PGM_RSRC1_GS, VGPRS);
			r1.priority        = KYTY_PM4_GET(value, SPI_SHADER_PGM_RSRC1_GS, PRIORITY);
			r1.float_mode      = KYTY_PM4_GET(value, SPI_SHADER_PGM_RSRC1_GS, FLOAT_MODE);
			r1.dx10_clamp      = KYTY_PM4_GET(value, SPI_SHADER_PGM_RSRC1_GS, DX10_CLAMP) != 0;
			r1.debug_mode      = KYTY_PM4_GET(value, SPI_SHADER_PGM_RSRC1_GS, DEBUG_MODE) != 0;
			r1.ieee_mode       = KYTY_PM4_GET(value, SPI_SHADER_PGM_RSRC1_GS, IEEE_MODE) != 0;
			r1.cu_group_enable = KYTY_PM4_GET(value, SPI_SHADER_PGM_RSRC1_GS, CU_GROUP_ENABLE) != 0;
			r1.require_forward_progress =
			    KYTY_PM4_GET(value, SPI_SHADER_PGM_RSRC1_GS, FWD_PROGRESS) != 0;
			r1.threadgroup_configuration =
			    KYTY_PM4_GET(value, SPI_SHADER_PGM_RSRC1_GS, WGP_MODE) != 0;
			r1.gs_vgpr_component_count =
			    KYTY_PM4_GET(value, SPI_SHADER_PGM_RSRC1_GS, GS_VGPR_COMP_CNT);
			r1.fp16_overflow = KYTY_PM4_GET(value, SPI_SHADER_PGM_RSRC1_GS, FP16_OVFL) != 0;
			vs.gs_regs.rsrc1 = r1;
			break;
		}
		case Pm4::SPI_SHADER_PGM_RSRC2_GS: {
			HW::GsShaderResource2 r2;
			r2.scratch_en = KYTY_PM4_GET(value, SPI_SHADER_PGM_RSRC2_GS, SCRATCH_EN) != 0;
			r2.user_sgpr  = KYTY_PM4_GET(value, SPI_SHADER_PGM_RSRC2_GS, USER_SGPR) +
			                (KYTY_PM4_GET(value, SPI_SHADER_PGM_RSRC2_GS, USER_SGPR_MSB) << 5u);
			r2.es_vgpr_component_count =
			    KYTY_PM4_GET(value, SPI_SHADER_PGM_RSRC2_GS, ES_VGPR_COMP_CNT);
			r2.offchip_lds   = KYTY_PM4_GET(value, SPI_SHADER_PGM_RSRC2_GS, OC_LDS_EN) != 0;
			r2.lds_size      = KYTY_PM4_GET(value, SPI_SHADER_PGM_RSRC2_GS, LDS_SIZE);
			r2.shared_vgprs  = KYTY_PM4_GET(value, SPI_SHADER_PGM_RSRC2_GS, SHARED_VGPR_CNT);
			vs.gs_regs.rsrc2 = r2;
			break;
		}
		case ComputeChecksum:
		case PixelChecksum:
		case GeometryChecksum: break;
		default: NoteUnhandled(state, "sh", offset); break;
	}
}

void SetUserConfigRegister(GuestState& state, uint32_t offset, uint32_t value) {
	switch (offset) {
		case Pm4::GE_CNTL: {
			HW::GeControl control;
			control.primitive_group_size = KYTY_PM4_GET(value, GE_CNTL, PRIM_GRP_SIZE);
			control.vertex_group_size    = KYTY_PM4_GET(value, GE_CNTL, VERT_GRP_SIZE);
			state.user_config.SetGeControl(control);
			break;
		}
		case Pm4::GE_USER_VGPR_EN: {
			HW::GeUserVgprEn enable;
			enable.vgpr1 = KYTY_PM4_GET(value, GE_USER_VGPR_EN, EN_USER_VGPR1) != 0;
			enable.vgpr2 = KYTY_PM4_GET(value, GE_USER_VGPR_EN, EN_USER_VGPR2) != 0;
			enable.vgpr3 = KYTY_PM4_GET(value, GE_USER_VGPR_EN, EN_USER_VGPR3) != 0;
			state.user_config.SetGeUserVgprEn(enable);
			break;
		}
		default: NoteUnhandled(state, "uc", offset); break;
	}
}

void ApplyHeaderRegisters(GuestState& state, const Shader& shader) {
	for (uint32_t i = 0; shader.cx_registers != nullptr && i < shader.num_cx_registers; i++) {
		SetContextRegister(state, shader.cx_registers[i].offset, shader.cx_registers[i].value);
	}
	for (uint32_t i = 0; shader.sh_registers != nullptr && i < shader.num_sh_registers; i++) {
		SetShRegister(state, shader.sh_registers[i].offset, shader.sh_registers[i].value);
	}
	if (shader.specials == nullptr) {
		return;
	}
	const auto& specials    = *shader.specials;
	state.dispatch_modifier = specials.dispatch_modifier;
	if (specials.ge_cntl.offset != 0) {
		SetUserConfigRegister(state, specials.ge_cntl.offset, specials.ge_cntl.value);
	}
	if (specials.vgt_shader_stages_en.offset != 0) {
		SetContextRegister(state, specials.vgt_shader_stages_en.offset,
		                   specials.vgt_shader_stages_en.value);
	}
	if (specials.vgt_gs_out_prim_type.offset != 0) {
		SetContextRegister(state, specials.vgt_gs_out_prim_type.offset,
		                   specials.vgt_gs_out_prim_type.value);
	}
	if (specials.ge_user_vgpr_en.offset != 0) {
		SetUserConfigRegister(state, specials.ge_user_vgpr_en.offset,
		                      specials.ge_user_vgpr_en.value);
	}
}


bool PrepareVertexTables(const Shader& shader, HW::VertexShaderInfo& vertex, VertexTables& tables,
                         std::string& note) {
	ShaderMappedData data;
	data.type                = static_cast<Prospero::ShaderBinaryType>(shader.type);
	data.user_data           = shader.user_data;
	data.input_semantics     = shader.input_semantics;
	data.num_input_semantics = shader.num_input_semantics;
	if (shader.input_semantics != nullptr) {
		std::copy_n(shader.input_semantics,
		            std::min<uint32_t>(shader.num_input_semantics, ShaderMappedData::MaxInputSemantics),
		            data.input_semantics_snapshot.begin());
	}
	data.code_size_bytes     = shader.shader_size;
	data.scratch_size_dwords = shader.scratch_size_dw_per_thread;
	ShaderVertexMetadata metadata;
	std::string          error;
	if (!ShaderReadVertexMetadata(data, HW::UserSgprInfo::SGPRS_MAX, metadata, &error)) {
		note = "vertex metadata: " + error;
		return false;
	}
	if (metadata.vertex_attrib_reg < 0) {
		return true;
	}
	// VertexAttribFormat k32Float, k32_32Float, k32_32_32Float, k32_32_32_32Float.
	static constexpr std::array<uint32_t, 5> float_formats = {311, 88, 257, 298, 311};
	for (const auto& semantic: metadata.input_semantics) {
		const auto format = float_formats[std::min<uint32_t>(semantic.size_in_elements, 4u)];
		tables.attrib[semantic.semantic] = (semantic.semantic & 0x1fu) | (format << 5u);
	}
	for (uint32_t i = 0; i < 32; i++) {
		const uint64_t base      = 0x100000000ull + i * 0x1000000ull;
		tables.buffer[i * 4 + 0] = static_cast<uint32_t>(base);
		tables.buffer[i * 4 + 1] = static_cast<uint32_t>(base >> 32u) | (16u << 16u);
		tables.buffer[i * 4 + 2] = 0x10000u;
		tables.buffer[i * 4 + 3] = DstSel(4, 5, 6, 7);
	}
	const auto set_pointer = [&](int reg, const void* table) {
		const auto address                = reinterpret_cast<uint64_t>(table);
		vertex.gs_user_sgpr.value[reg]     = static_cast<uint32_t>(address);
		vertex.gs_user_sgpr.value[reg + 1] = static_cast<uint32_t>(address >> 32u);
	};
	set_pointer(metadata.vertex_attrib_reg, tables.attrib.data());
	set_pointer(metadata.vertex_buffer_reg, tables.buffer.data());
	note = fmt::format("stand-in vertex tables ({} semantics)", metadata.input_semantics.size());
	return true;
}

// Stand-in guest memory and user data: one 16-byte pattern laid over every address and over the
// user SGPRs. Read as a buffer descriptor (V#, 16-byte aligned) it is a raw 2 GB buffer; as an
// image descriptor (T#, 32-byte aligned) a null image, as dword 0 is zero; as a pointer, an
// address below 2^48 that reads the same pattern. Materialization then succeeds with unbound
// images, as when a game draws with nothing bound to a slot.
constexpr std::array<uint32_t, 4> MemoryPattern = {0u, 0u, 0x7ffffff0u, DstSel(4, 5, 6, 7)};

bool ReadPatternMemory(void* /*userdata*/, uint64_t address, std::span<uint32_t> values) {
	for (size_t i = 0; i < values.size(); i++) {
		values[i] = MemoryPattern[((address >> 2u) + i) & 3u];
	}
	return true;
}

void FillUserData(HW::UserSgprInfo& user_data) {
	for (uint32_t i = 0; i < HW::UserSgprInfo::SGPRS_MAX; i++) {
		user_data.value[i] = MemoryPattern[i & 3u];
	}
}


void AppendNote(std::string& note, const std::string& text) {
	if (!text.empty()) {
		note += (note.empty() ? "" : "; ") + text;
	}
}

// ---------------------------------------------------------------------------------------------
// Extraction: where the shaders are in a game's own files. The AGC header block in eboot.bin is
// the universal half (every PS5 title has one, so nothing here knows an engine), the CSDR bundle
// the FromSoftware/Bluepoint half. Ports of tools/local/static-precompile/agc.py.
// ---------------------------------------------------------------------------------------------

// struct Shader: magic '1234', version 0x18, then 64-bit self-relative pointers (a stored value is
// relative to the pointer field's own address; 0 is null):
//   0 u32 header   4 u32 version   8 user_data  16 code (0 in files)  24 cx_registers
//  32 sh_registers  40 specials  48 input_semantics  56 output_semantics
//  64 u32 header_size  68 u32 shader_size  72 u32 embedded_cb_dqw  76 u32 target
//  80 u32 num_input_semantics  84 u16 scratch_dw  86 u16 num_output_semantics
//  88 u16 special_sizes_bytes  90 u8 type  91 u8 n_cx  92 u8 n_sh
constexpr std::array<uint8_t, 8> HeaderMagic = {'1', '2', '3', '4', 0x18, 0x00, 0x00, 0x00};

// A structurally invalid file: the caller skips it and says so.
class ExtractError : public std::runtime_error {
public:
	using std::runtime_error::runtime_error;
};

std::vector<uint8_t> ReadBytes(const std::filesystem::path& path) {
	std::ifstream file(path, std::ios::binary);
	return {std::istreambuf_iterator<char>(file), std::istreambuf_iterator<char>()};
}

uint32_t U32(std::span<const uint8_t> b, uint64_t o) {
	uint32_t v = 0;
	if (o + 4 > b.size()) {
		throw ExtractError("read past the end of the image");
	}
	std::memcpy(&v, b.data() + o, 4);
	return v;
}

uint16_t U16(std::span<const uint8_t> b, uint64_t o) {
	uint16_t v = 0;
	if (o + 2 > b.size()) {
		throw ExtractError("read past the end of the image");
	}
	std::memcpy(&v, b.data() + o, 2);
	return v;
}

uint64_t U64(std::span<const uint8_t> b, uint64_t o) {
	uint64_t v = 0;
	if (o + 8 > b.size()) {
		throw ExtractError("read past the end of the image");
	}
	std::memcpy(&v, b.data() + o, 8);
	return v;
}

int64_t I64(std::span<const uint8_t> b, uint64_t o) {
	int64_t v = 0;
	if (o + 8 > b.size()) {
		throw ExtractError("read past the end of the image");
	}
	std::memcpy(&v, b.data() + o, 8);
	return v;
}

bool IsMagic(std::span<const uint8_t> b, uint64_t o, std::span<const uint8_t> magic) {
	return o + magic.size() <= b.size() &&
	       std::memcmp(b.data() + o, magic.data(), magic.size()) == 0;
}

// Elf64_Phdr: type u32, flags u32, offset, vaddr, paddr, filesz, memsz, align.
struct Phdr {
	uint32_t type;
	uint64_t offset;
	uint64_t vaddr;
	uint64_t filesz;
};

Phdr ReadPhdr(std::span<const uint8_t> elf, uint64_t phoff, uint64_t ent, uint64_t i) {
	const auto at = phoff + ent * i;
	return {.type = U32(elf, at), .offset = U64(elf, at + 8), .vaddr = U64(elf, at + 16),
	        .filesz = U64(elf, at + 32)};
}

std::vector<Phdr> ElfLoads(std::span<const uint8_t> elf) {
	if (elf.size() < 64 || std::memcmp(elf.data(), "\x7f"
	                                              "ELF",
	                                    4) != 0) {
		throw ExtractError("not an ELF file");
	}
	const auto        phoff     = U64(elf, 32);
	const auto        phentsize = U16(elf, 54);
	const auto        phnum     = U16(elf, 56);
	std::vector<Phdr> loads;
	for (uint64_t i = 0; i < phnum; i++) {
		loads.push_back(ReadPhdr(elf, phoff, phentsize, i));
	}
	return loads;
}

// The ELF image of an eboot.bin: the file when it is one, else the ELF of its SELF with plaintext
// segments (a fake-signed one, as the emulator's loader reads it: src/loader/elf.cpp). Each SELF
// segment flagged 0x800 holds the file bytes of program header (type >> 20) & 0xfff; the image
// puts them back at that header's file offset.
std::vector<uint8_t> PlainElf(const std::vector<uint8_t>& file) {
	if (file.size() >= 4 && std::memcmp(file.data(), "\x7f"
	                                              "ELF",
	                                    4) == 0) {
		return file;
	}
	// SELF magics: SCE's two variants.
	const std::array<std::array<uint8_t, 4>, 2> self_magics = {
	    std::array<uint8_t, 4> {0x4f, 0x15, 0x3d, 0x1d},
	    std::array<uint8_t, 4> {0x54, 0x14, 0xf5, 0xee}};
	const bool is_self = file.size() >= 32 &&
	                     std::any_of(self_magics.begin(), self_magics.end(),
	                                 [&](const auto& magic) {
		                                 return std::memcmp(file.data(), magic.data(), 4) == 0;
	                                 });
	if (!is_self) {
		throw ExtractError("not an ELF file nor a SELF");
	}
	const auto     file_size = U64(file, 16);
	const auto     segments  = U16(file, 24);
	const uint64_t ehdr      = 32 + 32 * segments;
	if (file.size() < ehdr + 64 || std::memcmp(file.data() + ehdr, "\x7f"
	                                                                 "ELF",
	                                            4) != 0) {
		throw ExtractError("SELF without an ELF header");
	}
	const std::span<const uint8_t> inner(file.data() + ehdr, file.size() - ehdr);
	const auto                     phoff     = U64(inner, 32);
	const auto                     phentsize = U16(inner, 54);
	const auto                     phnum     = U16(inner, 56);
	const auto                     table_end = phoff + phnum * phentsize;
	uint64_t                       size      = table_end;
	for (uint64_t i = 0; i < phnum; i++) {
		const auto p = ReadPhdr(inner, phoff, phentsize, i);
		size         = std::max(size, p.offset + p.filesz);
	}
	std::vector<uint8_t> image(size, 0);
	if (table_end > size || table_end > inner.size()) {
		throw ExtractError("SELF program header table outside the image");
	}
	std::memcpy(image.data(), inner.data(), table_end);
	for (uint64_t i = 0; i < segments; i++) {
		const auto kind   = U64(file, 32 + 32 * i);
		const auto at     = U64(file, 32 + 32 * i + 8);
		const auto stored = U64(file, 32 + 32 * i + 16);
		const auto count  = U64(file, 32 + 32 * i + 24);
		const auto index  = (kind >> 20) & 0xfffu;
		if ((kind & 0x800u) == 0 || index >= phnum) {
			continue;
		}
		const auto p = ReadPhdr(inner, phoff, phentsize, index);
		if ((kind & 0x2u) != 0 || stored != count || count != p.filesz ||
		    at + count > file.size()) {
			throw ExtractError("an encrypted or compressed SELF: the emulator needs the decrypted "
			                   "game (its plain eboot.bin or a fake-signed SELF)");
		}
		std::memcpy(image.data() + p.offset, file.data() + at, count);
	}
	// A header whose bytes no segment holds: what follows the SELF's own bytes, when they are its size.
	for (uint64_t i = 0; i < phnum; i++) {
		const auto p = ReadPhdr(inner, phoff, phentsize, i);
		if (p.filesz == 0 || p.filesz != file.size() - file_size || p.offset + p.filesz > size) {
			continue;
		}
		const std::span<const uint8_t> held(image.data() + p.offset, p.filesz);
		if (std::all_of(held.begin(), held.end(), [](uint8_t b) { return b == 0; })) {
			std::memcpy(image.data() + p.offset, file.data() + file_size, p.filesz);
		}
	}
	return image;
}

// R_X86_64_RELATIVE relocations of an ELF image, as target VA -> addend. The pairs that point a
// header and its code into the same table are what locate the code of an embedded shader.
std::map<uint64_t, int64_t> RelativeRelocations(std::span<const uint8_t> elf,
                                                const std::vector<Phdr>&   loads) {
	std::map<uint64_t, int64_t> relocations;
	const auto dynamic = std::find_if(loads.begin(), loads.end(),
	                                  [](const Phdr& p) { return p.type == 2; }); // PT_DYNAMIC
	if (dynamic == loads.end()) {
		return relocations;
	}
	std::map<int64_t, uint64_t> tags;
	for (uint64_t i = 0; i < dynamic->filesz / 16; i++) {
		const auto tag = I64(elf, dynamic->offset + 16 * i);
		if (tag == 0) {
			break;
		}
		tags[tag] = U64(elf, dynamic->offset + 16 * i + 8);
	}
	if (!tags.contains(7) || !tags.contains(8)) { // DT_RELA, DT_RELASZ
		return relocations;
	}
	const auto va_to_off = [&](uint64_t va) -> std::optional<uint64_t> {
		for (const auto& p : loads) {
			if (p.type == 1 && p.vaddr <= va && va < p.vaddr + p.filesz) { // PT_LOAD
				return p.offset + va - p.vaddr;
			}
		}
		return std::nullopt;
	};
	const auto start = va_to_off(tags[7]);
	if (!start.has_value()) {
		return relocations;
	}
	for (uint64_t i = 0; i < tags[8] / 24; i++) {
		const auto where = U64(elf, *start + 24 * i);
		const auto info  = U64(elf, *start + 24 * i + 8);
		if ((info & 0xffffffffu) == 8) { // R_X86_64_RELATIVE
			relocations[where] = I64(elf, *start + 24 * i + 16);
		}
	}
	return relocations;
}

// The AGC headers in eboot.bin's data segment with their code. The code of header k starts where
// header k-1's (256-byte aligned) code ends, and a relocated (header, code) pointer pair in the
// game's shader table fixes where the sequence starts.
std::vector<ShaderBlob> EmbeddedShaders(const std::filesystem::path& game) {
	auto path = game / "decrypted" / "eboot.bin";
	if (!std::filesystem::is_regular_file(path)) {
		path = game / "eboot.bin";
	}
	if (!std::filesystem::is_regular_file(path)) {
		return {};
	}
	const auto elf   = PlainElf(ReadBytes(path));
	const auto loads = ElfLoads(elf);
	const auto first = std::search(elf.begin(), elf.end(), HeaderMagic.begin(), HeaderMagic.end());
	if (first == elf.end()) {
		return {};
	}
	const auto first_at = static_cast<uint64_t>(first - elf.begin());
	const auto segment  = std::find_if(loads.begin(), loads.end(), [&](const Phdr& p) {
		return p.type == 1 && p.offset <= first_at && first_at < p.offset + p.filesz;
	});
	if (segment == loads.end()) {
		throw ExtractError("the header block is not in a loadable segment");
	}
	const auto delta = segment->offset - segment->vaddr; // file offset - VA, in the data segment
	const auto limit = segment->offset + segment->filesz;
	std::vector<std::pair<uint64_t, std::vector<uint8_t>>> headers;
	for (uint64_t pos = first_at; pos < limit;) {
		const auto size = U32(elf, pos + 64);
		if (size < 96 || pos + size > elf.size()) {
			throw ExtractError(fmt::format("implausible header_size {} at {:#x}", size, pos));
		}
		headers.emplace_back(pos, std::vector<uint8_t>(elf.begin() + pos,
		                                              elf.begin() + pos + size));
		const auto next = pos + size;
		if (!IsMagic(elf, next, HeaderMagic)) {
			break;
		}
		pos = next;
	}
	std::map<uint64_t, size_t> header_va;
	std::vector<uint32_t>      sizes;
	for (size_t k = 0; k < headers.size(); k++) {
		header_va[headers[k].first - delta] = k;
		sizes.push_back(U32(headers[k].second, 68));
	}
	const auto relocations = RelativeRelocations(elf, loads);
	std::optional<uint64_t> starts;
	for (const auto& [where, addend]: relocations) {
		if (addend < 0) {
			continue;
		}
		const auto k = header_va.find(static_cast<uint64_t>(addend));
		if (k == header_va.end() || !relocations.contains(where + 8)) {
			continue;
		}
		auto code_va = static_cast<uint64_t>(relocations.at(where + 8));
		for (size_t j = 0; j < k->second; j++) {
			code_va -= (sizes[j] + 0xffu) & ~0xffu;
		}
		starts = code_va;
		break;
	}
	if (!starts.has_value()) {
		throw ExtractError("no relocation pairs an embedded header with its code");
	}
	std::vector<ShaderBlob> out;
	uint64_t                code_va = *starts;
	for (size_t k = 0; k < headers.size(); k++) {
		const auto off  = code_va + delta;
		const auto size = sizes[k];
		if (off + size > elf.size()) {
			throw ExtractError("a shader's code is outside the image");
		}
		out.push_back({.header = std::vector<char>(headers[k].second.begin(),
		                                          headers[k].second.end()),
		               .code   = std::vector<char>(elf.begin() + off, elf.begin() + off + size),
		               .source = fmt::format("eboot:{:#x}", headers[k].first)});
		code_va = (code_va + size + 0xffu) & ~0xffu;
	}
	return out;
}

// A CSDR bundle: a table of entries, each the shader code followed by its AGC header.
std::vector<ShaderBlob> ParseBundle(std::span<const uint8_t> data, const std::string& label) {
	if (data.size() < 20) {
		throw ExtractError("truncated bundle header");
	}
	if (std::memcmp(data.data() + data.size() - 12, "RDSC", 4) != 0 ||
	    U32(data, data.size() - 4) != data.size() - 12) {
		throw ExtractError("invalid CSDR footer or payload length");
	}
	const auto count = U32(data, 4);
	if (count == 0 || count > 16384 || 8u + static_cast<uint64_t>(count) * 20u > data.size()) {
		throw ExtractError("unsupported or truncated entry table");
	}
	const uint64_t table_end = 8u + static_cast<uint64_t>(count) * 20u;
	const uint64_t aligned   = (table_end + 7u) & ~7u;
	const std::span<const uint8_t> padding(data.data() + table_end, aligned - table_end);
	if (!std::all_of(padding.begin(), padding.end(), [](uint8_t b) { return b == 0xff; })) {
		throw ExtractError("invalid table alignment padding");
	}
	std::vector<ShaderBlob> entries;
	uint64_t                cursor = aligned;
	for (uint32_t i = 0; i < count; i++) {
		const uint64_t entry        = 8u + static_cast<uint64_t>(i) * 20u;
		const auto     code_bytes   = U32(data, entry + 8);
		const auto     header_bytes = U32(data, entry + 12);
		const auto     relative     = U32(data, entry + 16);
		const uint64_t begin        = entry + 16 + relative;
		const uint64_t header       = begin + code_bytes;
		const uint64_t end          = header + header_bytes;
		if (begin != cursor || code_bytes == 0 || code_bytes % 4 != 0 || header_bytes < 96 ||
		    end > data.size() - 12) {
			throw ExtractError(fmt::format("entry {}: non-contiguous or out-of-bounds payload", i));
		}
		if (std::memcmp(data.data() + header, "1234", 4) != 0) {
			throw ExtractError(fmt::format("entry {}: missing AGC header", i));
		}
		const auto saved_header = U32(data, header + 64);
		const auto saved_code   = U32(data, header + 68);
		if (saved_code != code_bytes || saved_header != header_bytes || data[header + 90] > 8) {
			throw ExtractError(fmt::format("entry {}: inconsistent AGC sizes or type", i));
		}
		entries.push_back({.header = std::vector<char>(data.begin() + header, data.begin() + end),
		                   .code   = std::vector<char>(data.begin() + begin, data.begin() + header),
		                   .source = fmt::format("{}#{}", label, i)});
		cursor = end;
	}
	if (cursor != data.size() - 12) {
		throw ExtractError("unrecognized trailing bundle data");
	}
	return entries;
}

// Every shader the game's own files hold: eboot.bin first, then one sorted pass over its CSDR
// bundles (the '_trinity' twins are the PS5 Pro variants).
uint64_t ShaderHash(const ShaderBlob& blob) {
	return XXH3_64bits(blob.code.data(), blob.code.size());
}

std::vector<ShaderBlob> EnumerateGame(const std::filesystem::path& game, std::string& report) {
	std::vector<ShaderBlob> shaders;
	try {
		auto embedded = EmbeddedShaders(game);
		report += fmt::format("  eboot.bin: {} shaders\n", embedded.size());
		shaders.insert(shaders.end(), std::make_move_iterator(embedded.begin()),
		               std::make_move_iterator(embedded.end()));
	} catch (const ExtractError& error) {
		report += fmt::format("  eboot.bin: {}\n", error.what());
	}
	std::vector<std::filesystem::path> bundles;
	std::error_code                    code;
	for (std::filesystem::recursive_directory_iterator it(game, code), end; it != end;
	     it.increment(code)) {
		if (code) {
			break;
		}
		if (it->is_regular_file(code) && it->path().extension() == ".csdr" &&
		    !it->path().stem().string().ends_with("_trinity")) {
			bundles.push_back(it->path());
		}
	}
	std::sort(bundles.begin(), bundles.end());
	const auto before = shaders.size();
	size_t     loaded = 0;
	size_t     failed = 0;
	for (const auto& path: bundles) {
		try {
			auto entries = ParseBundle(ReadBytes(path), path.filename().string());
			loaded++;
			shaders.insert(shaders.end(), std::make_move_iterator(entries.begin()),
			               std::make_move_iterator(entries.end()));
		} catch (const ExtractError& error) {
			failed++;
			if (failed <= 8) {
				report += fmt::format("  {}: {}\n", path.filename().string(), error.what());
			}
		}
	}
	if (!bundles.empty()) {
		report += fmt::format("  csdr: {}/{} bundles, {} shaders\n", loaded, bundles.size(),
		                      shaders.size() - before);
	}
	return shaders;
}

// A folder of <hash>.header + <hash>.code pairs, the form the runtime dump and the batch tool share.
// Only the extensions matter: the manifest is for the batch tool's driver, not for us.
std::vector<ShaderBlob> LoadCorpus(const std::filesystem::path& folder, std::string& report) {
	std::vector<std::filesystem::path> headers;
	std::error_code                    code;
	for (std::filesystem::directory_iterator it(folder, code), end; it != end; it.increment(code)) {
		if (code) {
			break;
		}
		if (it->is_regular_file(code) && it->path().extension() == ".header") {
			headers.push_back(it->path());
		}
	}
	std::sort(headers.begin(), headers.end());
	std::vector<ShaderBlob> out;
	size_t                  unreadable = 0;
	out.reserve(headers.size());
	for (const auto& path: headers) {
		auto       header = ReadBytes(path);
		const auto body   = ReadBytes(std::filesystem::path(path).replace_extension(".code"));
		if (header.size() < 96 || body.empty()) {
			unreadable++;
			continue;
		}
		out.push_back({.header = std::vector<char>(header.begin(), header.end()),
		               .code   = std::vector<char>(body.begin(), body.end()),
		               .source = fmt::format("corpus:{}", path.stem().string())});
	}
	report += fmt::format("  corpus {}: {} shaders{}\n", folder.filename().string(), out.size(),
	                      unreadable == 0 ? "" : fmt::format(", {} unreadable", unreadable));
	return out;
}

// The draw's state the header does not hold gets the same defaults the batch tool has always used:
// the input primitive that fits the header's GE_CNTL group size, identity pixel interpolators, the
// default export mapping and stand-in vertex tables. A seeded module therefore matches the draws
// whose own state is those defaults; every other draw misses and compiles as it does today.
Status PrepareOfflineProgram(const ShaderBlob& blob, Program& program, std::string& note,
                             std::string& failure) {
	auto* header = CopyAligned(blob.header, 16);
	auto* code   = CopyAligned(blob.code, 256);

	Shader* shader = nullptr;
	if (Gen5::AgcCreateShader(&shader, header, code) != 0 || shader == nullptr) {
		failure = "AgcCreateShader failed";
		return Status::Failed;
	}

	GuestState state;
	ApplyHeaderRegisters(state, *shader);
	FillUserData(state.compute.cs_user_sgpr);
	FillUserData(state.pixel.ps_user_sgpr);
	FillUserData(state.vertex.gs_user_sgpr);
	AppendNote(note, state.unhandled.empty() ? "" : "unhandled registers " + state.unhandled);

	auto& params        = program.params;
	auto& compute_info  = program.compute;
	auto& pixel_info    = program.pixel;
	auto& vertex_info   = program.vertex;
	auto& vertex_tables = program.vertex_tables;
	auto& input         = program.input;

	switch (static_cast<Prospero::ShaderBinaryType>(shader->type)) {
		case Prospero::ShaderBinaryType::kCs:
			state.compute.cs_regs.wave_size  = Pm4::ComputeWaveSize(state.dispatch_modifier);
			state.compute.cs_user_sgpr.count = state.compute.cs_regs.user_sgpr;
			compute_info.host_subgroup_size  = 32;
			params = PrepareProgram(state.compute, state.context.GetShaderRegisters(), compute_info);
			program.stage     = ShaderType::Compute;
			program.wave_size = compute_info.wave_size;
			input.compute     = &compute_info;
			break;
		case Prospero::ShaderBinaryType::kPs: {
			for (uint32_t i = 0; i < 32; i++) {
				state.context.SetPsInputSettings(i, i);
			}
			const std::array<Prospero::ColorComponentMapping, 8> export_mapping {};
			params = PrepareProgram(state.pixel, state.context.GetShaderRegisters(), export_mapping,
			                        pixel_info);
			program.stage     = ShaderType::Pixel;
			program.wave_size = pixel_info.wave_size;
			input.pixel       = &pixel_info;
			break;
		}
		case Prospero::ShaderBinaryType::kGs: {
			// The draw's input primitive, as the header's GE_CNTL groups it: meshlet shaders take
			// one point per subgroup (VERT_GRP_SIZE 1) and expand it.
			const auto vertex_group = state.user_config.GetGeControl().vertex_group_size;
			state.user_config.SetPrimitiveType(vertex_group == 1   ? Prospero::PrimitiveType::kPointList
			                                   : vertex_group == 2 ? Prospero::PrimitiveType::kLineList
			                                                       : Prospero::PrimitiveType::kTriList);
			const bool merged = (state.context.GetShaderStages() & 0x20u) != 0;
			if (!merged) {
				std::string table_note;
				if (!PrepareVertexTables(*shader, state.vertex, vertex_tables, table_note)) {
					failure = table_note;
					return Status::Failed;
				}
				AppendNote(note, table_note);
			}
			params = PrepareProgram(state.vertex, state.context, state.user_config, vertex_info);
			program.stage = vertex_info.logical_stage;
			if (program.stage == ShaderType::Mesh) {
				auto& mesh              = vertex_info.mesh;
				mesh.host_subgroup_size = 32;
				program.wave_size       = mesh.wave_size;
				// The host limits of the RTX 5090 (VK_EXT_mesh_shader), as GetGraphicsPrograms
				// applies them: a larger subgroup runs in passes, and a draw that still does not fit
				// is skipped.
				mesh.passes = mesh.PassesFor(128u);
				if (mesh.passes > 1u) {
					AppendNote(note, fmt::format("mesh passes={} (wave{} threads={})", mesh.passes,
					                             mesh.wave_size, mesh.HostThreads()));
				}
				if (mesh.passes == 0u || mesh.max_vertices > 256u || mesh.max_primitives > 256u ||
				    mesh.lds_size_dwords * 4u > 28672u) {
					AppendNote(note, fmt::format("mesh exceeds host limits: wave{} threads={} "
					                             "vertices={} primitives={} LDS={}",
					                             mesh.wave_size, mesh.HostThreads(),
					                             mesh.max_vertices, mesh.max_primitives,
					                             mesh.lds_size_dwords));
				}
			} else {
				program.user_data_base = 8;
				program.wave_size      = vertex_info.wave_size;
			}
			input.vertex = &vertex_info;
			break;
		}
		default:
			// The precompile reports this verbatim, and a count of skips nobody can act on is not a
			// report: name the type so a title whose whole vertex path is fetch shaders is visible.
			failure = fmt::format("type {}: a fetch shader or a fused half, nothing to compile alone",
			                      shader->type);
			return Status::NotStandalone;
	}
	return Status::Ok;
}
} // namespace Libs::Graphics::Offline
