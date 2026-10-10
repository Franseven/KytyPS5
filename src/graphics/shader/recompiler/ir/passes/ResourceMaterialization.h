#ifndef EMULATOR_INCLUDE_EMULATOR_GRAPHICS_SHADER_RECOMPILER_RESOURCEMATERIALIZATION_H_
#define EMULATOR_INCLUDE_EMULATOR_GRAPHICS_SHADER_RECOMPILER_RESOURCEMATERIALIZATION_H_

#include "graphics/shader/recompiler/ir/passes/SrtWalker.h"

namespace Libs::Graphics::ShaderRecompiler::IR {

// Canonical module-affecting resource state. Runtime addresses and descriptor payloads remain in
// ResourceSnapshot and therefore do not create shader permutations.
// Serialized by the shader disk cache (shaderDiskCache.cpp, Visit): a new field goes there too.
struct ResourceSpecialization {
	struct Buffer {
		uint32_t               packed_stride                   = 0;
		Prospero::BufferFormat descriptor_format               = Prospero::BufferFormat::kInvalid;
		uint32_t               descriptor_swizzle              = DstSel(4, 5, 6, 7);
		bool                   zero_stride_oob                 = false;
		uint32_t               indirect_root                   = BufferResource::NoIndirectBuffer;
		uint32_t               indirect_mapping_offset         = 0;
		uint32_t               indirect_search_iterations      = 0;
		bool                   operator==(const Buffer&) const = default;
	};

	struct Image {
		Prospero::TextureNumericClass numeric_class = Prospero::TextureNumericClass::Unsupported;
		Decoder::ImageDimension       dimension     = Decoder::ImageDimension::Unknown;
		uint32_t                      mip_count     = 1;
		Prospero::BufferFormat        conversion_format          = Prospero::BufferFormat::kInvalid;
		uint32_t                      shader_swizzle             = ShaderImageIdentitySwizzle;
		uint32_t                      indirect_root              = ImageResource::NoIndirectImage;
		uint32_t                      indirect_mapping_offset    = 0;
		uint32_t                      indirect_search_iterations = 0;
		bool                          cube                       = false;
		bool                          fmask                      = false;
		bool                          bindless                   = false;
		bool                          operator==(const Image&) const = default;
	};

	struct Sampler {
		bool     bindless                = false;
		uint32_t bindless_mapping_offset = 0;
		bool     operator==(const Sampler&) const = default;
	};

	std::vector<Buffer>  buffers;
	std::vector<Image>   images;
	std::vector<Sampler> samplers;

	bool operator==(const ResourceSpecialization&) const = default;
};

// Extracts the descriptor/SRT value graph before resource specialization. The returned plan owns
// its values and is independent of the translated shader CFG.
ResourcePlan ExtractResourcePlan(const Program& program);

// Refreshes cached resources and specialization in place. A failed refresh must not be used.
bool MaterializeResources(const ResourcePlan& program, const SrtRuntime& runtime,
                          ResourceSnapshot& snapshot, ResourceSpecialization& specialization);

// Diagnostics: the source line of the indirect-image check that failed the last
// MaterializeResources call on this thread, or 0.
int LastIndirectImageFailureLine();

// Where a refresh spends its microseconds. Il walk SRT e' il 28% del frame su GT7 e la domanda e'
// *cosa* dentro di lui: costruire i due walker e la sessione di replay (una volta per draw), il
// refresh dei flat slot, o i tre giri di descrittori. Sono tre fix diversi -- il primo si toglie
// riusando i walker, il secondo e' il walker, il terzo e' il numero di descrittori valutati.
// Accumulati per thread e **prelevati con azzeramento** (vedi TakeSrtPhaseTimings): chi chiama
// MaterializeResources una volta per draw legge il delta di quel draw.
struct SrtPhaseTimings {
	uint64_t setup_us   = 0; // costruzione dei due walker + sessione di replay, per chiamata
	uint64_t flat_us    = 0; // RefreshFlatBuffer: i flat slot
	uint64_t buffers_us = 0; // il giro dei buffer, write extent compresi
	uint64_t images_us  = 0; // il giro delle immagini
	uint64_t rest_us    = 0; // sampler, user data, canonicalizzazione
};

// Il delta accumulato su questo thread dall'ultima chiamata, e azzera.
SrtPhaseTimings TakeSrtPhaseTimings();

// The buffer word (BufferWord) of a materialized V#: what a portable module reads instead of the
// stride and format the specialization baked. 0 when the descriptor is not a buffer one.
uint32_t BufferDescriptorWord(const DescriptorValue& descriptor);

// A buffer format's layout and component type as a buffer word holds them (bits 0-6 of the word
// shifted right by BufferWord::LayoutShift); 0 for a format without them.
uint32_t BufferFormatClass(Prospero::BufferFormat format);

// Whether the runtime format decode covers a format: true for packed formats and byte-aligned ones
// with 4-byte components, the classes the emitted decode was bisected onto. A false here keeps the
// format specialized (one module per format), which PortableFormats is the only caller of.
bool PortableFormatClass(Prospero::BufferFormat format);

// The portable form of a specialization (PortableShaders): the formatted buffers the code only
// loads from decode the format of their buffer words (RuntimeBufferFormat) instead of the V#'s, so
// one module serves every format. False when that leaves the specialization unchanged.
bool PortableFormats(const ShaderInfo& info, ResourceSpecialization& specialization);

// Merges the specializations the module provably cannot tell apart, so a record made before the
// merge canonicalizes to the permutation the game now selects. The plan is what says whether a
// buffer's stride reaches the emitted code at all.
void CanonicalizeSpecialization(const ResourcePlan& program, ResourceSpecialization& specialization);

// Applies an already-derived specialization to native IR before layout and emission.
void ApplyResourceSpecialization(Program& program, const ResourceSpecialization& specialization);

} // namespace Libs::Graphics::ShaderRecompiler::IR

#endif /* EMULATOR_INCLUDE_EMULATOR_GRAPHICS_SHADER_RECOMPILER_RESOURCEMATERIALIZATION_H_ */
