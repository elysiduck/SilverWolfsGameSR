#pragma once

#include <cstdint>
#include <memory>
#include <string_view>

struct ID3D11Texture2D;

namespace Magpie {

class DeviceResources;

// Runs AMD SESR-M7 INT8 x2 on an XDNA2 NPU. The public interface deliberately
// contains no Ryzen AI types so ARM64 and non-SESR builds can still parse it.
class SesrNpuProcessor {
public:
	static constexpr std::string_view EFFECT_NAME = "SESR-M7 NPU x2";
	static constexpr std::string_view FAST_EFFECT_NAME = "SESR-M7 NPU x2 Fast";
	static constexpr std::string_view EXACT_1080P_EFFECT_NAME =
		"SESR-M7 NPU Exact 1080p";
	// Explicit diagnostic path: bypass the Exact-1080p pre-resize and submit
	// the captured texture at its current dimensions.
	static constexpr std::string_view CURRENT_INPUT_EFFECT_NAME =
		"SESR-M7 NPU Current Input";
	static bool IsSesrEffect(std::string_view name) noexcept;
	// When the selected ModelHub model is a full-frame GameSR residual core,
	// returns its required exact input size (for v1.7.1 this is 1280x720).
	// Legacy tiled SESR models return false and keep the old x2 sizing rule.
	static bool TryGetExactInputSize(std::wstring_view modelName, uint32_t& width, uint32_t& height) noexcept;

	SesrNpuProcessor() noexcept;
	~SesrNpuProcessor() noexcept;

	SesrNpuProcessor(const SesrNpuProcessor&) = delete;
	SesrNpuProcessor& operator=(const SesrNpuProcessor&) = delete;

	bool Initialize(
		DeviceResources& deviceResources,
		ID3D11Texture2D* input,
		bool fastMode = false,
		std::wstring_view modelName = {}) noexcept;
	bool Process(ID3D11Texture2D* input) noexcept;

	ID3D11Texture2D* GetOutput() const noexcept;

private:
	struct Impl;
	std::unique_ptr<Impl> _impl;
};

}
