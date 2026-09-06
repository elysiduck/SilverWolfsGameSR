#include "pch.h"
#include "SesrNpuProcessor.h"

#include "DeviceResources.h"
#include "DirectXHelper.h"
#include "Logger.h"

#include <array>
#include <fstream>
#include <onnxruntime_cxx_api.h>

namespace Magpie {

namespace {

std::filesystem::path GetApplicationDirectory() {
	std::wstring path(32768, L'\0');
	const DWORD length = GetModuleFileNameW(nullptr, path.data(), (DWORD)path.size());
	if (!length || length >= path.size()) {
		return std::filesystem::current_path();
	}
	path.resize(length);
	return std::filesystem::path(path).parent_path();
}


std::wstring Utf8ToWide(std::string_view value) {
	if (value.empty()) {
		return {};
	}
	const int count = MultiByteToWideChar(
		CP_UTF8, MB_ERR_INVALID_CHARS, value.data(), (int)value.size(), nullptr, 0);
	if (count <= 0) {
		return {};
	}
	std::wstring result((size_t)count, L'\0');
	MultiByteToWideChar(
		CP_UTF8, MB_ERR_INVALID_CHARS, value.data(), (int)value.size(), result.data(), count);
	return result;
}

std::string ShapeToString(const std::vector<int64_t>& shape) {
	std::string result;
	for (size_t i = 0; i < shape.size(); ++i) {
		if (i) {
			result += "x";
		}
		result += std::to_string(shape[i]);
	}
	return result;
}

std::string TrimAscii(std::string value) {
	while (!value.empty() && (value.back() == '\r' || value.back() == '\n' ||
		value.back() == ' ' || value.back() == '\t')) {
		value.pop_back();
	}
	size_t begin = 0;
	while (begin < value.size() && (value[begin] == ' ' || value[begin] == '\t')) {
		++begin;
	}
	if (begin) {
		value.erase(0, begin);
	}
	return value;
}

uint64_t HashFileFNV1a64(const std::filesystem::path& path) {
	std::ifstream stream(path, std::ios::binary);
	if (!stream) {
		throw std::runtime_error("无法读取 GameSR ONNX 文件用于生成缓存键");
	}
	uint64_t hash = 14695981039346656037ull;
	std::array<char, 64 * 1024> buffer{};
	while (stream) {
		stream.read(buffer.data(), buffer.size());
		const std::streamsize count = stream.gcount();
		for (std::streamsize i = 0; i < count; ++i) {
			hash ^= (uint8_t)buffer[(size_t)i];
			hash *= 1099511628211ull;
		}
	}
	return hash;
}

struct SesrModelSelection {
	std::filesystem::path modelPath;
	std::filesystem::path cacheDir;
	std::string cacheKey;
	bool external = false;
};

SesrModelSelection ResolveSesrModel(
	const std::filesystem::path& applicationDir,
	const std::filesystem::path& sesrDir,
	std::wstring_view selectedModelName
) {
	const std::filesystem::path builtInModel =
		sesrDir / L"sesr_nhwc_int8_256x256.onnx";
	if (selectedModelName.empty()) {
		return {
			.modelPath = builtInModel,
			.cacheDir = sesrDir,
			.cacheKey = "modelcachekey_sesr_nhwc_int8_256x256",
			.external = false
		};
	}

	const std::filesystem::path modelsDir = applicationDir / L"models" / L"GameSR";
	const std::filesystem::path cacheDir = applicationDir / L"cache" / L"GameSR";
	std::error_code ec;
	std::filesystem::create_directories(modelsDir, ec);
	std::filesystem::create_directories(cacheDir, ec);

	// Presets store only a filename. Strip any directory component so a preset
	// can never escape Magpie's model library.
	const std::filesystem::path fileName = std::filesystem::path(std::wstring(selectedModelName)).filename();
	const std::filesystem::path selected = modelsDir / fileName;
	if (_wcsicmp(selected.extension().c_str(), L".onnx") != 0 ||
		!std::filesystem::is_regular_file(selected, ec)) {
		throw std::runtime_error(fmt::format(
			"预设绑定的 GameSR 模型不存在: models\\GameSR\\{}",
			selected.filename().string()));
	}

	const uint64_t hash = HashFileFNV1a64(selected);
	Logger::Get().Info(fmt::format(
		"GameSR 预设模型: {} (cache {:016x})",
		selected.filename().string(), hash));
	return {
		.modelPath = selected,
		.cacheDir = cacheDir,
		.cacheKey = fmt::format("modelcachekey_gamesr_{:016x}", hash),
		.external = true
	};
}

bool ProbeFullFrameGameSrShape(
	const std::filesystem::path& modelPath,
	uint32_t& width,
	uint32_t& height
) {
	Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "Magpie-GameSR-Probe");
	Ort::SessionOptions options;
	options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_DISABLE_ALL);
	options.SetIntraOpNumThreads(1);
	options.SetInterOpNumThreads(1);
	Ort::Session session(env, modelPath.c_str(), options);
	if (session.GetInputCount() != 1 || session.GetOutputCount() != 1) {
		return false;
	}
	const auto inputInfo = session.GetInputTypeInfo(0).GetTensorTypeAndShapeInfo();
	const auto outputInfo = session.GetOutputTypeInfo(0).GetTensorTypeAndShapeInfo();
	if (inputInfo.GetElementType() != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT ||
		outputInfo.GetElementType() != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT) {
		return false;
	}
	const std::vector<int64_t> in = inputInfo.GetShape();
	const std::vector<int64_t> out = outputInfo.GetShape();
	if (in.size() != 4 || out.size() != 4 ||
		in[0] != 1 || in[1] != 3 || out[0] != 1 || out[1] != 3 ||
		in[2] <= 0 || in[3] <= 0 || out[2] != in[2] || out[3] != in[3]) {
		return false;
	}
	width = (uint32_t)in[3];
	height = (uint32_t)in[2];
	return width > 0 && height > 0;
}

size_t ReflectIndex(ptrdiff_t index, size_t length) noexcept {
	if (length <= 1) {
		return 0;
	}
	const ptrdiff_t period = (ptrdiff_t)(2 * (length - 1));
	index %= period;
	if (index < 0) {
		index += period;
	}
	if ((size_t)index >= length) {
		index = period - index;
	}
	return (size_t)index;
}

bool IsExpectedShape(
	const std::vector<int64_t>& actual,
	const std::array<int64_t, 4>& expected
) noexcept {
	return actual.size() == expected.size() &&
		std::equal(actual.begin(), actual.end(), expected.begin());
}

std::string GetAsciiEnvironmentVariable(
	const char* name,
	std::string_view fallback
) {
	const DWORD length = GetEnvironmentVariableA(name, nullptr, 0);
	if (length <= 1) {
		return std::string(fallback);
	}
	std::string value(length, '\0');
	GetEnvironmentVariableA(name, value.data(), length);
	value.resize(length - 1);
	return value;
}

void ORT_API_CALL OrtLogCallback(
	void*,
	OrtLoggingLevel severity,
	const char*,
	const char*,
	const char*,
	const char* message
) noexcept {
	if (!message) {
		return;
	}

	const std::string_view text(message);
	const bool relevant = severity >= ORT_LOGGING_LEVEL_WARNING ||
		text.find("Vitis") != std::string_view::npos ||
		text.find("NPU") != std::string_view::npos ||
		text.find("Node placement") != std::string_view::npos ||
		text.find("assigned to") != std::string_view::npos ||
		text.find("VITIS_EP_CPU") != std::string_view::npos ||
		text.find("No. of Operators") != std::string_view::npos ||
		text.find("No. of Subgraphs") != std::string_view::npos;
	if (!relevant) {
		return;
	}

	try {
		const std::string formatted = fmt::format("ORT: {}", text);
		if (severity >= ORT_LOGGING_LEVEL_ERROR) {
			Logger::Get().Error(formatted);
		} else if (severity >= ORT_LOGGING_LEVEL_WARNING) {
			Logger::Get().Warn(formatted);
		} else {
			Logger::Get().Info(formatted);
		}
	} catch (...) {
	}
}

}

struct SesrNpuProcessor::Impl {
	static constexpr size_t TILE_SIZE = 256;
	static constexpr size_t SCALE = 2;
	static constexpr size_t OUTPUT_TILE_SIZE = TILE_SIZE * SCALE;
	static constexpr size_t CHANNELS = 3;
	static constexpr size_t INPUT_TILE_BYTES = TILE_SIZE * TILE_SIZE * CHANNELS;
	static constexpr size_t OUTPUT_TILE_BYTES =
		OUTPUT_TILE_SIZE * OUTPUT_TILE_SIZE * CHANNELS;

	DeviceResources* deviceResources = nullptr;
	uint32_t sourceWidth = 0;
	uint32_t sourceHeight = 0;
	uint32_t outputWidth = 0;
	uint32_t outputHeight = 0;
	uint32_t tilesX = 0;
	uint32_t tilesY = 0;
	uint32_t tileCount = 0;
	bool fastMode = false;
	size_t tileOverlap = 16;
	size_t coreSize = TILE_SIZE - 32;
	size_t outputOverlap = 32;
	size_t outputCoreSize = (TILE_SIZE - 32) * SCALE;
	DXGI_FORMAT sourceFormat = DXGI_FORMAT_UNKNOWN;

	enum class ModelMode {
		LegacyTiledInt8,
		GameSrResidualNchwF32
	};
	ModelMode modelMode = ModelMode::LegacyTiledInt8;

	// Full-frame GameSR v1.7.1 path. The NPU returns a 720p RGB residual.
	// We upload that residual once and let a compute shader perform both
	// half-pixel bilinear 1.5x sampling and residual composition on the GPU.
	std::vector<float> gameSrInput;
	std::vector<float> gameSrResidual;
	std::array<int64_t, 4> gameSrShape{ 1, 3, 0, 0 };
	winrt::com_ptr<ID3D11Texture2D> gameSrResidualTexture;
	winrt::com_ptr<ID3D11ShaderResourceView> gameSrResidualSrv;
	winrt::com_ptr<ID3D11UnorderedAccessView> gameSrOutputUav;
	winrt::com_ptr<ID3D11ComputeShader> gameSrComposeShader;
	winrt::com_ptr<ID3D11SamplerState> gameSrLinearSampler;

	struct ReadbackSlot {
		winrt::com_ptr<ID3D11Texture2D> texture;
		// Full-frame GameSR additionally keeps a GPU copy of the exact same
		// frame so base and NPU residual never come from different frames.
		winrt::com_ptr<ID3D11Texture2D> gpuTexture;
		winrt::com_ptr<ID3D11ShaderResourceView> gpuSrv;
		winrt::com_ptr<ID3D11Query> readyQuery;
		uint64_t frameId = 0;
		bool pending = false;
	};

	static constexpr size_t READBACK_SLOTS = 3;
	std::array<ReadbackSlot, READBACK_SLOTS> readbackSlots;
	size_t nextReadbackSlot = 0;
	uint64_t submittedReadbackFrames = 0;
	uint64_t droppedReadbackFrames = 0;
	uint64_t reusedOutputFrames = 0;

	winrt::com_ptr<ID3D11Texture2D> outputTexture;

	std::unique_ptr<Ort::Env> ortEnv;
	std::unique_ptr<Ort::Session> ortSession;
	std::unique_ptr<Ort::MemoryInfo> ortMemoryInfo;
	std::string inputName;
	std::string outputName;
	std::array<int64_t, 4> inputShape{ 1, TILE_SIZE, TILE_SIZE, CHANNELS };
	std::array<int64_t, 4> outputShape{ 1, OUTPUT_TILE_SIZE, OUTPUT_TILE_SIZE, CHANNELS };
	std::vector<int8_t> tileInputs;
	std::vector<int8_t> tileOutputs;
	std::vector<uint16_t> sourceXMap;
	std::vector<uint16_t> sourceYMap;
	std::vector<Ort::Value> inputTensors;
	std::vector<Ort::Value> outputTensors;
	uint64_t frameCount = 0;

	bool Initialize(
		DeviceResources& resources,
		ID3D11Texture2D* input,
		bool useFastMode,
		std::wstring_view modelName
	) {
#if !defined(_M_X64)
		(void)resources;
		(void)input;
		Logger::Get().Error("SESR-M7 NPU 仅支持 x64 Ryzen AI 设备");
		return false;
#else
		deviceResources = &resources;
		fastMode = useFastMode;
		tileOverlap = fastMode ? 0 : 16;
		coreSize = TILE_SIZE - 2 * tileOverlap;
		outputOverlap = tileOverlap * SCALE;
		outputCoreSize = coreSize * SCALE;

		D3D11_TEXTURE2D_DESC sourceDesc{};
		input->GetDesc(&sourceDesc);
		if (sourceDesc.Format != DXGI_FORMAT_B8G8R8A8_UNORM &&
			sourceDesc.Format != DXGI_FORMAT_R8G8B8A8_UNORM) {
			Logger::Get().Error(fmt::format(
				"SESR 不支持捕获纹理格式 {}", (uint32_t)sourceDesc.Format));
			return false;
		}
		if (!sourceDesc.Width || !sourceDesc.Height ||
			sourceDesc.Width > D3D11_REQ_TEXTURE2D_U_OR_V_DIMENSION / SCALE ||
			sourceDesc.Height > D3D11_REQ_TEXTURE2D_U_OR_V_DIMENSION / SCALE) {
			Logger::Get().Error("SESR 输入尺寸无效");
			return false;
		}

		sourceWidth = sourceDesc.Width;
		sourceHeight = sourceDesc.Height;
		outputWidth = sourceWidth * SCALE;
		outputHeight = sourceHeight * SCALE;
		sourceFormat = sourceDesc.Format;
		tilesX = (sourceWidth + (uint32_t)coreSize - 1) / (uint32_t)coreSize;
		tilesY = (sourceHeight + (uint32_t)coreSize - 1) / (uint32_t)coreSize;
		tileCount = tilesX * tilesY;

		D3D11_TEXTURE2D_DESC stagingDesc = sourceDesc;
		stagingDesc.MipLevels = 1;
		stagingDesc.ArraySize = 1;
		stagingDesc.SampleDesc = { 1, 0 };
		stagingDesc.Usage = D3D11_USAGE_STAGING;
		stagingDesc.BindFlags = 0;
		stagingDesc.CPUAccessFlags = D3D11_CPU_ACCESS_READ;
		stagingDesc.MiscFlags = 0;
		D3D11_QUERY_DESC queryDesc{ D3D11_QUERY_EVENT, 0 };
		HRESULT hr = S_OK;
		for (ReadbackSlot& slot : readbackSlots) {
			hr = resources.GetD3DDevice()->CreateTexture2D(
				&stagingDesc, nullptr, slot.texture.put());
			if (FAILED(hr)) {
				Logger::Get().ComError("创建 SESR 异步读回纹理失败", hr);
				return false;
			}
			hr = resources.GetD3DDevice()->CreateQuery(&queryDesc, slot.readyQuery.put());
			if (FAILED(hr)) {
				Logger::Get().ComError("创建 SESR 异步读回查询失败", hr);
				return false;
			}
		}

		const D3D11_TEXTURE2D_DESC outputDesc{
			.Width = outputWidth,
			.Height = outputHeight,
			.MipLevels = 1,
			.ArraySize = 1,
			.Format = DXGI_FORMAT_R8G8B8A8_UNORM,
			.SampleDesc = { 1, 0 },
			.Usage = D3D11_USAGE_DYNAMIC,
			.BindFlags = D3D11_BIND_SHADER_RESOURCE,
			.CPUAccessFlags = D3D11_CPU_ACCESS_WRITE,
			.MiscFlags = 0
		};
		hr = resources.GetD3DDevice()->CreateTexture2D(
			&outputDesc, nullptr, outputTexture.put());
		if (FAILED(hr)) {
			Logger::Get().ComError("创建 SESR 输出纹理失败", hr);
			return false;
		}

		const std::filesystem::path applicationDir = GetApplicationDirectory();
		const std::filesystem::path sesrDir = applicationDir / L"sesr";
		const SesrModelSelection modelSelection =
			ResolveSesrModel(applicationDir, sesrDir, modelName);
		const std::filesystem::path& modelPath = modelSelection.modelPath;
		const std::filesystem::path xclbinPath =
			sesrDir / L"AMD_AIE2P_4x4_Overlay.xclbin";
		const std::filesystem::path configPath =
			sesrDir / L"vaip_config_stride.json";
		if (!std::filesystem::is_regular_file(modelPath) ||
			!std::filesystem::is_regular_file(xclbinPath) ||
			!std::filesystem::is_regular_file(configPath)) {
			Logger::Get().Error(fmt::format(
				"找不到 SESR 模型、VAIP 配置或 4x4 NPU overlay，请检查 {}",
				sesrDir.string()));
			return false;
		}
		const std::string target = GetAsciiEnvironmentVariable(
			"MAGPIE_SESR_TARGET", "X1");
		const std::string cacheKey = GetAsciiEnvironmentVariable(
			"MAGPIE_SESR_CACHE_KEY", modelSelection.cacheKey);
		const std::filesystem::path cacheDir = modelSelection.external
			? modelSelection.cacheDir : sesrDir;

		Logger::Get().Info(fmt::format(
			"SESR ONNX 阶段 1/4: 模型 {}，来源 {}，ORT {}，模式 {}，输入 {}x{}，切片数 {}，目标 {}，缓存 {}",
			modelPath.filename().string(),
			modelSelection.external ? "GameSR 预设" : "内置 SESR",
			OrtGetApiBase()->GetVersionString(),
			fastMode ? "快速（无重叠）" : "画质（16px 重叠）",
			sourceWidth, sourceHeight, tileCount, target, cacheKey));
		Logger::Get().Flush();

		SetEnvironmentVariableW(L"XLNX_ENABLE_CACHE", L"1");
		ortEnv = std::make_unique<Ort::Env>(
			ORT_LOGGING_LEVEL_VERBOSE, "Magpie-SESR", OrtLogCallback, nullptr);
		Ort::SessionOptions sessionOptions;
		sessionOptions.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
		sessionOptions.SetIntraOpNumThreads(1);
		sessionOptions.SetInterOpNumThreads(1);
		sessionOptions.SetLogSeverityLevel(ORT_LOGGING_LEVEL_VERBOSE);

		const std::unordered_map<std::string, std::string> providerOptions{
			{ "cache_dir", cacheDir.string() },
			{ "cache_key", cacheKey },
			{ "config_file", configPath.string() },
			{ "dd_use_elf_flow", "1" },
			{ "enable_preemption", "1" },
			{ "enable_cache_file_io_in_mem", "0" },
			{ "enable_txn_elf", "1" },
			{ "is_preemptible", "1" },
			{ "target", target },
			{ "xclbin", xclbinPath.string() },
			{ "xlnx_enable_old_qdq", "0" },
			{ "xlnx_enable_py3_round", "1" }
		};
		Logger::Get().Info(
			"SESR ONNX 阶段 2/4: 正在注册 VitisAIExecutionProvider");
		Logger::Get().Flush();
		sessionOptions.AppendExecutionProvider_VitisAI(providerOptions);

		Logger::Get().Info(fmt::format(
			"SESR ONNX 阶段 3/4: 正在加载模型 {}；{}缓存键 {}",
			modelPath.filename().string(),
			modelSelection.external ? "外部模型使用独立" : "内置模型使用预编译",
			cacheKey));
		Logger::Get().Flush();
		ortSession = std::make_unique<Ort::Session>(
			*ortEnv, modelPath.c_str(), sessionOptions);

		if (ortSession->GetInputCount() != 1 || ortSession->GetOutputCount() != 1) {
			Logger::Get().Error("SESR ONNX 模型必须只有一个输入和一个输出");
			return false;
		}

		const auto inputInfo =
			ortSession->GetInputTypeInfo(0).GetTensorTypeAndShapeInfo();
		const auto outputInfo =
			ortSession->GetOutputTypeInfo(0).GetTensorTypeAndShapeInfo();
		const std::vector<int64_t> actualInputShape = inputInfo.GetShape();
		const std::vector<int64_t> actualOutputShape = outputInfo.GetShape();
		const bool isLegacyTiled =
			inputInfo.GetElementType() == ONNX_TENSOR_ELEMENT_DATA_TYPE_INT8 &&
			outputInfo.GetElementType() == ONNX_TENSOR_ELEMENT_DATA_TYPE_INT8 &&
			IsExpectedShape(actualInputShape, inputShape) &&
			IsExpectedShape(actualOutputShape, outputShape);
		const bool isGameSrResidual =
			inputInfo.GetElementType() == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT &&
			outputInfo.GetElementType() == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT &&
			actualInputShape.size() == 4 && actualOutputShape.size() == 4 &&
			actualInputShape[0] == 1 && actualInputShape[1] == 3 &&
			actualOutputShape[0] == 1 && actualOutputShape[1] == 3 &&
			actualInputShape[2] == (int64_t)sourceHeight &&
			actualInputShape[3] == (int64_t)sourceWidth &&
			actualOutputShape[2] == actualInputShape[2] &&
			actualOutputShape[3] == actualInputShape[3];

		if (!isLegacyTiled && !isGameSrResidual) {
			Logger::Get().Error(fmt::format(
				"模型 {} 接口不受支持。支持：① INT8 NHWC [1,256,256,3]->[1,512,512,3] SESR；② FP32 NCHW [1,3,H,W]->同尺寸 residual GameSR。当前输入 {}，输出 {}。",
				modelPath.filename().string(),
				ShapeToString(actualInputShape),
				ShapeToString(actualOutputShape)));
			return false;
		}

		Ort::AllocatorWithDefaultOptions allocator;
		auto allocatedInputName = ortSession->GetInputNameAllocated(0, allocator);
		auto allocatedOutputName = ortSession->GetOutputNameAllocated(0, allocator);
		inputName = allocatedInputName.get();
		outputName = allocatedOutputName.get();
		ortMemoryInfo = std::make_unique<Ort::MemoryInfo>(
			Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault));

		if (isLegacyTiled) {
			modelMode = ModelMode::LegacyTiledInt8;
			tileInputs.resize((size_t)tileCount * INPUT_TILE_BYTES);
			tileOutputs.resize((size_t)tileCount * OUTPUT_TILE_BYTES);

			// Precompute the reflected source coordinates once. The old hot loop
			// evaluated ReflectIndex() several times for every pixel of every tile.
			sourceXMap.resize((size_t)tilesX * TILE_SIZE);
			sourceYMap.resize((size_t)tilesY * TILE_SIZE);
			const size_t paddedWidth = (size_t)tilesX * coreSize;
			const size_t paddedHeight = (size_t)tilesY * coreSize;
			for (uint32_t tileX = 0; tileX < tilesX; ++tileX) {
				for (size_t x = 0; x < TILE_SIZE; ++x) {
					const ptrdiff_t paddedX =
						(ptrdiff_t)tileX * coreSize + (ptrdiff_t)x - (ptrdiff_t)tileOverlap;
					sourceXMap[(size_t)tileX * TILE_SIZE + x] = (uint16_t)ReflectIndex(
						(ptrdiff_t)ReflectIndex(paddedX, paddedWidth), sourceWidth);
				}
			}
			for (uint32_t tileY = 0; tileY < tilesY; ++tileY) {
				for (size_t y = 0; y < TILE_SIZE; ++y) {
					const ptrdiff_t paddedY =
						(ptrdiff_t)tileY * coreSize + (ptrdiff_t)y - (ptrdiff_t)tileOverlap;
					sourceYMap[(size_t)tileY * TILE_SIZE + y] = (uint16_t)ReflectIndex(
						(ptrdiff_t)ReflectIndex(paddedY, paddedHeight), sourceHeight);
				}
			}

			inputTensors.reserve(tileCount);
			outputTensors.reserve(tileCount);
			for (uint32_t tileIndex = 0; tileIndex < tileCount; ++tileIndex) {
				inputTensors.emplace_back(Ort::Value::CreateTensor<int8_t>(
					*ortMemoryInfo,
					tileInputs.data() + (size_t)tileIndex * INPUT_TILE_BYTES,
					INPUT_TILE_BYTES,
					inputShape.data(),
					inputShape.size()));
				outputTensors.emplace_back(Ort::Value::CreateTensor<int8_t>(
					*ortMemoryInfo,
					tileOutputs.data() + (size_t)tileIndex * OUTPUT_TILE_BYTES,
					OUTPUT_TILE_BYTES,
					outputShape.data(),
					outputShape.size()));
			}

			Logger::Get().Info(fmt::format(
				"SESR ONNX 阶段 4/4: legacy tiled NPU 会话完成，{}x{} -> {}x{}，{} 个切片",
				sourceWidth, sourceHeight, outputWidth, outputHeight, tileCount));
		} else {
			modelMode = ModelMode::GameSrResidualNchwF32;
			// v1.7.1 contract: residual is LR, final image is a 1.5x half-pixel
			// bilinear base plus a 1.5x half-pixel bilinear residual.
			outputWidth = (sourceWidth * 3) / 2;
			outputHeight = (sourceHeight * 3) / 2;
			gameSrShape = {
				1, 3, (int64_t)sourceHeight, (int64_t)sourceWidth
			};
			const size_t plane = (size_t)sourceWidth * sourceHeight;
			gameSrInput.resize(plane * 3);
			gameSrResidual.resize(plane * 3);
			inputTensors.clear();
			outputTensors.clear();
			inputTensors.emplace_back(Ort::Value::CreateTensor<float>(
				*ortMemoryInfo, gameSrInput.data(), gameSrInput.size(),
				gameSrShape.data(), gameSrShape.size()));
			outputTensors.emplace_back(Ort::Value::CreateTensor<float>(
				*ortMemoryInfo, gameSrResidual.data(), gameSrResidual.size(),
				gameSrShape.data(), gameSrShape.size()));

			// Replace the legacy CPU-writable x2 output with a GPU UAV at the
			// actual GameSR output resolution.
			outputTexture = nullptr;
			D3D11_TEXTURE2D_DESC gameSrOutputDesc{};
			gameSrOutputDesc.Width = outputWidth;
			gameSrOutputDesc.Height = outputHeight;
			gameSrOutputDesc.MipLevels = 1;
			gameSrOutputDesc.ArraySize = 1;
			gameSrOutputDesc.Format = DXGI_FORMAT_R8G8B8A8_UNORM;
			gameSrOutputDesc.SampleDesc = { 1, 0 };
			gameSrOutputDesc.Usage = D3D11_USAGE_DEFAULT;
			gameSrOutputDesc.BindFlags = D3D11_BIND_SHADER_RESOURCE | D3D11_BIND_UNORDERED_ACCESS;
			hr = resources.GetD3DDevice()->CreateTexture2D(
				&gameSrOutputDesc, nullptr, outputTexture.put());
			if (FAILED(hr)) {
				Logger::Get().ComError("创建 GameSR GPU 输出纹理失败", hr);
				return false;
			}
			hr = resources.GetD3DDevice()->CreateUnorderedAccessView(
				outputTexture.get(), nullptr, gameSrOutputUav.put());
			if (FAILED(hr)) {
				Logger::Get().ComError("创建 GameSR 输出 UAV 失败", hr);
				return false;
			}

			D3D11_TEXTURE2D_DESC residualDesc{};
			residualDesc.Width = sourceWidth;
			residualDesc.Height = sourceHeight;
			residualDesc.MipLevels = 1;
			residualDesc.ArraySize = 1;
			residualDesc.Format = DXGI_FORMAT_R32G32B32A32_FLOAT;
			residualDesc.SampleDesc = { 1, 0 };
			residualDesc.Usage = D3D11_USAGE_DYNAMIC;
			residualDesc.BindFlags = D3D11_BIND_SHADER_RESOURCE;
			residualDesc.CPUAccessFlags = D3D11_CPU_ACCESS_WRITE;
			hr = resources.GetD3DDevice()->CreateTexture2D(
				&residualDesc, nullptr, gameSrResidualTexture.put());
			if (FAILED(hr)) {
				Logger::Get().ComError("创建 GameSR residual 上传纹理失败", hr);
				return false;
			}
			hr = resources.GetD3DDevice()->CreateShaderResourceView(
				gameSrResidualTexture.get(), nullptr, gameSrResidualSrv.put());
			if (FAILED(hr)) {
				Logger::Get().ComError("创建 GameSR residual SRV 失败", hr);
				return false;
			}
			D3D11_TEXTURE2D_DESC gpuFrameDesc = sourceDesc;
			gpuFrameDesc.MipLevels = 1;
			gpuFrameDesc.ArraySize = 1;
			gpuFrameDesc.SampleDesc = { 1, 0 };
			gpuFrameDesc.Usage = D3D11_USAGE_DEFAULT;
			gpuFrameDesc.BindFlags = D3D11_BIND_SHADER_RESOURCE;
			gpuFrameDesc.CPUAccessFlags = 0;
			gpuFrameDesc.MiscFlags = 0;
			for (ReadbackSlot& slot : readbackSlots) {
				hr = resources.GetD3DDevice()->CreateTexture2D(
					&gpuFrameDesc, nullptr, slot.gpuTexture.put());
				if (FAILED(hr)) {
					Logger::Get().ComError("创建 GameSR 帧匹配 GPU 纹理失败", hr);
					return false;
				}
				hr = resources.GetD3DDevice()->CreateShaderResourceView(
					slot.gpuTexture.get(), nullptr, slot.gpuSrv.put());
				if (FAILED(hr)) {
					Logger::Get().ComError("创建 GameSR 帧匹配 SRV 失败", hr);
					return false;
				}
			}

			D3D11_SAMPLER_DESC samplerDesc{};
			samplerDesc.Filter = D3D11_FILTER_MIN_MAG_MIP_LINEAR;
			samplerDesc.AddressU = D3D11_TEXTURE_ADDRESS_CLAMP;
			samplerDesc.AddressV = D3D11_TEXTURE_ADDRESS_CLAMP;
			samplerDesc.AddressW = D3D11_TEXTURE_ADDRESS_CLAMP;
			samplerDesc.MaxLOD = D3D11_FLOAT32_MAX;
			hr = resources.GetD3DDevice()->CreateSamplerState(
				&samplerDesc, gameSrLinearSampler.put());
			if (FAILED(hr)) {
				Logger::Get().ComError("创建 GameSR 线性采样器失败", hr);
				return false;
			}

			static constexpr std::string_view composeHlsl = R"(
Texture2D<float4> BaseTex : register(t0);
Texture2D<float4> ResidualTex : register(t1);
SamplerState LinearSampler : register(s0);
RWTexture2D<float4> OutputTex : register(u0);
[numthreads(8, 8, 1)]
void main(uint3 id : SV_DispatchThreadID) {
    uint outW, outH;
    OutputTex.GetDimensions(outW, outH);
    if (id.x >= outW || id.y >= outH) return;
    float2 uv = (float2(id.xy) + 0.5) / float2(outW, outH);
    float3 base = BaseTex.SampleLevel(LinearSampler, uv, 0).rgb;
    float3 residual = ResidualTex.SampleLevel(LinearSampler, uv, 0).rgb;
    OutputTex[id.xy] = float4(saturate(base + residual), 1.0);
}
)";
			winrt::com_ptr<ID3DBlob> shaderBlob;
			if (!DirectXHelper::CompileComputeShader(
				composeHlsl, "main", shaderBlob.put(), "GameSRComposeCS")) {
				return false;
			}
			hr = resources.GetD3DDevice()->CreateComputeShader(
				shaderBlob->GetBufferPointer(), shaderBlob->GetBufferSize(), nullptr,
				gameSrComposeShader.put());
			if (FAILED(hr)) {
				Logger::Get().ComError("创建 GameSR GPU 合成着色器失败", hr);
				return false;
			}

			Logger::Get().Info(fmt::format(
				"GameSR ONNX 阶段 4/4: full-frame residual NPU 会话完成，{}x{} NCHW -> residual，GPU half-pixel 1.5x 合成为 {}x{}",
				sourceWidth, sourceHeight, outputWidth, outputHeight));
		}
		Logger::Get().Flush();
		return true;
#endif
	}

	bool Process(ID3D11Texture2D* input) {
#if !defined(_M_X64)
		(void)input;
		return false;
#else
		D3D11_TEXTURE2D_DESC inputDesc{};
		input->GetDesc(&inputDesc);
		if (inputDesc.Width != sourceWidth || inputDesc.Height != sourceHeight ||
			inputDesc.Format != sourceFormat) {
			Logger::Get().Error("SESR 输入纹理尺寸或格式发生变化");
			return false;
		}

		ID3D11DeviceContext4* context = deviceResources->GetD3DDC();
		const auto frameBegin = std::chrono::steady_clock::now();

		// Pipeline stage 1: submit the current GPU texture to the first free staging slot.
		// A busy slot is never overwritten. If every slot is still in flight, we skip only
		// the capture submission and continue to process any already-completed slot.
		ReadbackSlot* writeSlot = nullptr;
		for (size_t i = 0; i < READBACK_SLOTS; ++i) {
			ReadbackSlot& candidate = readbackSlots[(nextReadbackSlot + i) % READBACK_SLOTS];
			if (!candidate.pending) {
				writeSlot = &candidate;
				nextReadbackSlot = ((nextReadbackSlot + i + 1) % READBACK_SLOTS);
				break;
			}
			const HRESULT status = context->GetData(
				candidate.readyQuery.get(), nullptr, 0, D3D11_ASYNC_GETDATA_DONOTFLUSH);
			if (status == S_OK) {
				candidate.pending = false;
				writeSlot = &candidate;
				nextReadbackSlot = ((nextReadbackSlot + i + 1) % READBACK_SLOTS);
				break;
			}
		}

		if (writeSlot) {
			if (modelMode == ModelMode::GameSrResidualNchwF32) {
				context->CopyResource(writeSlot->gpuTexture.get(), input);
				context->CopyResource(writeSlot->texture.get(), writeSlot->gpuTexture.get());
			} else {
				context->CopyResource(writeSlot->texture.get(), input);
			}
			context->End(writeSlot->readyQuery.get());
			writeSlot->frameId = ++submittedReadbackFrames;
			writeSlot->pending = true;
		} else {
			++droppedReadbackFrames;
		}

		// Pipeline stage 2: process the newest completed capture, not necessarily the oldest.
		// This bounds queue latency and avoids processing stale frames after a GPU hiccup.
		ReadbackSlot* readySlot = nullptr;
		uint64_t newestFrameId = 0;
		for (ReadbackSlot& slot : readbackSlots) {
			if (!slot.pending) {
				continue;
			}
			const HRESULT status = context->GetData(
				slot.readyQuery.get(), nullptr, 0, D3D11_ASYNC_GETDATA_DONOTFLUSH);
			if (status == S_OK && slot.frameId >= newestFrameId) {
				readySlot = &slot;
				newestFrameId = slot.frameId;
			}
		}

		// The first frame must populate outputTexture. For subsequent frames, avoid
		// blocking the render thread just because the GPU copy is not ready yet.
		if (!readySlot && frameCount == 0 && writeSlot) {
			readySlot = writeSlot;
			newestFrameId = writeSlot->frameId;
		}

		if (!readySlot) {
			++reusedOutputFrames;
			return true;
		}

		// Any older completed captures are now stale because we only want the newest frame.
		for (ReadbackSlot& slot : readbackSlots) {
			if (&slot != readySlot && slot.pending && slot.frameId < readySlot->frameId) {
				const HRESULT status = context->GetData(
					slot.readyQuery.get(), nullptr, 0, D3D11_ASYNC_GETDATA_DONOTFLUSH);
				if (status == S_OK) {
					slot.pending = false;
					++droppedReadbackFrames;
				}
			}
		}

		const auto beforeMap = std::chrono::steady_clock::now();
		D3D11_MAPPED_SUBRESOURCE mappedInput{};
		HRESULT hr = context->Map(
			readySlot->texture.get(), 0, D3D11_MAP_READ, 0, &mappedInput);
		const auto afterMap = std::chrono::steady_clock::now();
		if (FAILED(hr)) {
			Logger::Get().ComError("读取 SESR 异步输入纹理失败", hr);
			return false;
		}

		if (modelMode == ModelMode::GameSrResidualNchwF32) {
			const auto packBegin = afterMap;
			const size_t plane = (size_t)sourceWidth * sourceHeight;
			constexpr float INV_255 = 1.0f / 255.0f;
			for (uint32_t y = 0; y < sourceHeight; ++y) {
				const uint8_t* row =
					(const uint8_t*)mappedInput.pData + (size_t)y * mappedInput.RowPitch;
				const size_t rowBase = (size_t)y * sourceWidth;
				for (uint32_t x = 0; x < sourceWidth; ++x) {
					const uint8_t* pixel = row + (size_t)x * 4;
					const float r = (sourceFormat == DXGI_FORMAT_B8G8R8A8_UNORM
						? pixel[2] : pixel[0]) * INV_255;
					const float g = pixel[1] * INV_255;
					const float b = (sourceFormat == DXGI_FORMAT_B8G8R8A8_UNORM
						? pixel[0] : pixel[2]) * INV_255;
					const size_t i = rowBase + x;
					gameSrInput[i] = r;
					gameSrInput[plane + i] = g;
					gameSrInput[plane * 2 + i] = b;
				}
			}
			context->Unmap(readySlot->texture.get(), 0);
			readySlot->pending = false;
			const auto packEnd = std::chrono::steady_clock::now();

			const char* inputNames[]{ inputName.c_str() };
			const char* outputNames[]{ outputName.c_str() };
			Ort::RunOptions runOptions;
			const auto npuBegin = packEnd;
			ortSession->Run(
				runOptions,
				inputNames,
				&inputTensors[0],
				1,
				outputNames,
				&outputTensors[0],
				1);
			const auto npuEnd = std::chrono::steady_clock::now();

			D3D11_MAPPED_SUBRESOURCE mappedResidual{};
			const auto uploadBegin = npuEnd;
			hr = context->Map(
				gameSrResidualTexture.get(), 0, D3D11_MAP_WRITE_DISCARD, 0,
				&mappedResidual);
			if (FAILED(hr)) {
				Logger::Get().ComError("上传 GameSR residual 到 GPU 失败", hr);
				return false;
			}
			for (uint32_t y = 0; y < sourceHeight; ++y) {
				float* row = (float*)((uint8_t*)mappedResidual.pData +
					(size_t)y * mappedResidual.RowPitch);
				const size_t rowBase = (size_t)y * sourceWidth;
				for (uint32_t x = 0; x < sourceWidth; ++x) {
					const size_t i = rowBase + x;
					row[(size_t)x * 4] = gameSrResidual[i];
					row[(size_t)x * 4 + 1] = gameSrResidual[plane + i];
					row[(size_t)x * 4 + 2] = gameSrResidual[plane * 2 + i];
					row[(size_t)x * 4 + 3] = 0.0f;
				}
			}
			context->Unmap(gameSrResidualTexture.get(), 0);
			const auto uploadEnd = std::chrono::steady_clock::now();

			// Compose on GPU using the GPU copy paired with this exact readback slot.
			// This preserves temporal alignment even though Pipeline v3 intentionally
			// processes the newest completed (not necessarily current) capture.
			ID3D11ShaderResourceView* srvs[]{
				readySlot->gpuSrv.get(), gameSrResidualSrv.get()
			};
			ID3D11SamplerState* samplers[]{ gameSrLinearSampler.get() };
			ID3D11UnorderedAccessView* uavs[]{ gameSrOutputUav.get() };
			context->CSSetShader(gameSrComposeShader.get(), nullptr, 0);
			context->CSSetShaderResources(0, 2, srvs);
			context->CSSetSamplers(0, 1, samplers);
			context->CSSetUnorderedAccessViews(0, 1, uavs, nullptr);
			context->Dispatch((outputWidth + 7) / 8, (outputHeight + 7) / 8, 1);
			ID3D11UnorderedAccessView* nullUav = nullptr;
			ID3D11ShaderResourceView* nullSrvs[2]{ nullptr, nullptr };
			context->CSSetUnorderedAccessViews(0, 1, &nullUav, nullptr);
			context->CSSetShaderResources(0, 2, nullSrvs);
			context->CSSetShader(nullptr, nullptr, 0);
			const auto composeSubmitEnd = std::chrono::steady_clock::now();

			++frameCount;
			if (frameCount == 1 || frameCount % 60 == 0) {
				const double wallMs = std::chrono::duration<double, std::milli>(
					composeSubmitEnd - frameBegin).count();
				const double mapMs = std::chrono::duration<double, std::milli>(
					afterMap - beforeMap).count();
				const double packMs = std::chrono::duration<double, std::milli>(
					packEnd - packBegin).count();
				const double npuMs = std::chrono::duration<double, std::milli>(
					npuEnd - npuBegin).count();
				const double uploadMs = std::chrono::duration<double, std::milli>(
					uploadEnd - uploadBegin).count();
				const double composeSubmitMs = std::chrono::duration<double, std::milli>(
					composeSubmitEnd - uploadEnd).count();
				Logger::Get().Info(fmt::format(
					"GameSR PIPELINE v3: Process墙钟 {:.2f} ms, input Map等待 {:.2f} ms, CPU NCHW打包 {:.2f} ms, NPU {:.2f} ms, residual上传 {:.2f} ms, GPU 1.5x+合成提交 {:.2f} ms, 处理帧 {}, 提交读回 {}, 丢弃读回 {}, 复用旧输出 {}",
					wallMs, mapMs, packMs, npuMs, uploadMs, composeSubmitMs,
					frameCount, submittedReadbackFrames, droppedReadbackFrames,
					reusedOutputFrames));
			}
			return true;
		}

		const auto sliceBegin = afterMap;
		for (uint32_t tileY = 0; tileY < tilesY; ++tileY) {
			for (uint32_t tileX = 0; tileX < tilesX; ++tileX) {
				const size_t tileIndex = (size_t)tileY * tilesX + tileX;
				int8_t* tile = tileInputs.data() + tileIndex * INPUT_TILE_BYTES;
				for (size_t y = 0; y < TILE_SIZE; ++y) {
					const size_t sourceY = sourceYMap[(size_t)tileY * TILE_SIZE + y];
					const uint8_t* sourceRow =
						(const uint8_t*)mappedInput.pData + sourceY * mappedInput.RowPitch;
					const size_t sourceXBase = (size_t)tileX * TILE_SIZE;
					for (size_t x = 0; x < TILE_SIZE; ++x) {
						const size_t sourceX = sourceXMap[sourceXBase + x];
						const uint8_t* pixel = sourceRow + sourceX * 4;
						const uint8_t r = sourceFormat == DXGI_FORMAT_B8G8R8A8_UNORM
							? pixel[2] : pixel[0];
						const uint8_t g = pixel[1];
						const uint8_t b = sourceFormat == DXGI_FORMAT_B8G8R8A8_UNORM
							? pixel[0] : pixel[2];
						const size_t offset = (y * TILE_SIZE + x) * CHANNELS;
						tile[offset] = (int8_t)((int)r - 128);
						tile[offset + 1] = (int8_t)((int)g - 128);
						tile[offset + 2] = (int8_t)((int)b - 128);
					}
				}
			}
		}
		context->Unmap(readySlot->texture.get(), 0);
		readySlot->pending = false;
		const auto sliceEnd = std::chrono::steady_clock::now();

		const char* inputNames[]{ inputName.c_str() };
		const char* outputNames[]{ outputName.c_str() };
		Ort::RunOptions runOptions;
		const auto npuBegin = sliceEnd;
		for (uint32_t tileIndex = 0; tileIndex < tileCount; ++tileIndex) {
			ortSession->Run(
				runOptions,
				inputNames,
				&inputTensors[tileIndex],
				1,
				outputNames,
				&outputTensors[tileIndex],
				1);
		}
		const auto npuEnd = std::chrono::steady_clock::now();

		D3D11_MAPPED_SUBRESOURCE mappedOutput{};
		const auto outputMapBegin = npuEnd;
		hr = context->Map(
			outputTexture.get(), 0, D3D11_MAP_WRITE_DISCARD, 0, &mappedOutput);
		const auto outputMapEnd = std::chrono::steady_clock::now();
		if (FAILED(hr)) {
			Logger::Get().ComError("写入 SESR 输出纹理失败", hr);
			return false;
		}

		// Fast merge: move invariant bounds/offset math out of the per-pixel loop.
		const auto mergeBegin = outputMapEnd;
		for (uint32_t tileY = 0; tileY < tilesY; ++tileY) {
			const size_t baseDestY = (size_t)tileY * outputCoreSize;
			if (baseDestY >= outputHeight) {
				break;
			}
			const size_t copyHeight = std::min(outputCoreSize, (size_t)outputHeight - baseDestY);
			for (uint32_t tileX = 0; tileX < tilesX; ++tileX) {
				const size_t baseDestX = (size_t)tileX * outputCoreSize;
				if (baseDestX >= outputWidth) {
					break;
				}
				const size_t copyWidth = std::min(outputCoreSize, (size_t)outputWidth - baseDestX);
				const size_t tileIndex = (size_t)tileY * tilesX + tileX;
				const int8_t* tile =
					tileOutputs.data() + tileIndex * OUTPUT_TILE_BYTES;
				for (size_t y = 0; y < copyHeight; ++y) {
					const int8_t* src = tile +
						(((y + outputOverlap) * OUTPUT_TILE_SIZE + outputOverlap) * CHANNELS);
					uint8_t* dst = (uint8_t*)mappedOutput.pData +
						(baseDestY + y) * mappedOutput.RowPitch + baseDestX * 4;
					for (size_t x = 0; x < copyWidth; ++x) {
						dst[0] = (uint8_t)((int)src[0] + 128);
						dst[1] = (uint8_t)((int)src[1] + 128);
						dst[2] = (uint8_t)((int)src[2] + 128);
						dst[3] = 255;
						src += CHANNELS;
						dst += 4;
					}
				}
			}
		}
		const auto mergeEnd = std::chrono::steady_clock::now();
		context->Unmap(outputTexture.get(), 0);
		const auto unmapEnd = std::chrono::steady_clock::now();

		++frameCount;
		if (frameCount == 1 || frameCount % 60 == 0) {
			const double processWallMs = std::chrono::duration<double, std::milli>(
				unmapEnd - frameBegin).count();
			const double mapMs = std::chrono::duration<double, std::milli>(
				afterMap - beforeMap).count();
			const double sliceMs = std::chrono::duration<double, std::milli>(
				sliceEnd - sliceBegin).count();
			const double npuMs = std::chrono::duration<double, std::milli>(
				npuEnd - npuBegin).count();
			const double outputMapMs = std::chrono::duration<double, std::milli>(
				outputMapEnd - outputMapBegin).count();
			const double mergeMs = std::chrono::duration<double, std::milli>(
				mergeEnd - mergeBegin).count();
			const double unmapMs = std::chrono::duration<double, std::milli>(
				unmapEnd - mergeEnd).count();
			const double totalMs = std::chrono::duration<double, std::milli>(
				unmapEnd - frameBegin).count();
			Logger::Get().Info(fmt::format(
				"SESR PIPELINE v3: Process墙钟 {:.2f} ms, input Map等待 {:.2f} ms, CPU切片 {:.2f} ms, NPU {:.2f} ms ({:.2f} ms/块), output Map {:.2f} ms, CPU拼接 {:.2f} ms, Unmap {:.2f} ms, 处理帧 {}, 提交读回 {}, 丢弃读回 {}, 复用旧输出 {}",
				processWallMs, mapMs, sliceMs, npuMs, npuMs / tileCount,
				outputMapMs, mergeMs, unmapMs, totalMs,
				frameCount, droppedReadbackFrames, reusedOutputFrames));
		}
		return true;
#endif
	}
};

SesrNpuProcessor::SesrNpuProcessor() noexcept = default;
SesrNpuProcessor::~SesrNpuProcessor() noexcept = default;

bool SesrNpuProcessor::Initialize(
	DeviceResources& deviceResources,
	ID3D11Texture2D* input,
	bool fastMode,
	std::wstring_view modelName
) noexcept {
	try {
		auto impl = std::make_unique<Impl>();
		if (!impl->Initialize(deviceResources, input, fastMode, modelName)) {
			return false;
		}
		_impl = std::move(impl);
		return true;
	} catch (const Ort::Exception& e) {
		Logger::Get().Error(fmt::format(
			"初始化 SESR ONNX NPU 失败 (ORT {}): {}",
			(int)e.GetOrtErrorCode(), e.what()));
		return false;
	} catch (const std::exception& e) {
		Logger::Get().Error(fmt::format(
			"初始化 SESR ONNX NPU 失败: {}", e.what()));
		return false;
	} catch (...) {
		Logger::Get().Error("初始化 SESR ONNX NPU 失败: 未知错误");
		return false;
	}
}

bool SesrNpuProcessor::IsSesrEffect(std::string_view name) noexcept {
	return name == EFFECT_NAME || name == FAST_EFFECT_NAME ||
		name == EXACT_1080P_EFFECT_NAME;
}

bool SesrNpuProcessor::TryGetExactInputSize(
	std::wstring_view modelName,
	uint32_t& width,
	uint32_t& height
) noexcept {
	width = 0;
	height = 0;
#if !defined(_M_X64)
	return false;
#else
	try {
		const std::filesystem::path applicationDir = GetApplicationDirectory();
		const std::filesystem::path sesrDir = applicationDir / L"sesr";
		const SesrModelSelection selection = ResolveSesrModel(applicationDir, sesrDir, modelName);
		if (!selection.external) {
			return false;
		}
		if (!ProbeFullFrameGameSrShape(selection.modelPath, width, height)) {
			return false;
		}
		Logger::Get().Info(fmt::format(
			"GameSR 预设: 模型 {} 请求 Exact 输入 {}x{}",
			selection.modelPath.filename().string(), width, height));
		return true;
	} catch (const Ort::Exception& e) {
		Logger::Get().Warn(fmt::format(
			"GameSR 预设: 探测外部模型尺寸失败 (ORT {}): {}。将暂用 legacy Exact 尺寸规则。",
			(int)e.GetOrtErrorCode(), e.what()));
		return false;
	} catch (const std::exception& e) {
		Logger::Get().Warn(fmt::format(
			"GameSR 预设: 探测外部模型尺寸失败: {}。将暂用 legacy Exact 尺寸规则。",
			e.what()));
		return false;
	} catch (...) {
		Logger::Get().Warn(
			"GameSR 预设: 探测外部模型尺寸失败。将暂用 legacy Exact 尺寸规则。");
		return false;
	}
#endif
}

bool SesrNpuProcessor::Process(ID3D11Texture2D* input) noexcept {
	if (!_impl) {
		return false;
	}
	try {
		return _impl->Process(input);
	} catch (const Ort::Exception& e) {
		Logger::Get().Error(fmt::format(
			"SESR ONNX NPU 处理失败 (ORT {}): {}",
			(int)e.GetOrtErrorCode(), e.what()));
		return false;
	} catch (const std::exception& e) {
		Logger::Get().Error(fmt::format(
			"SESR ONNX NPU 处理失败: {}", e.what()));
		return false;
	} catch (...) {
		Logger::Get().Error("SESR ONNX NPU 处理失败: 未知错误");
		return false;
	}
}

ID3D11Texture2D* SesrNpuProcessor::GetOutput() const noexcept {
	return _impl ? _impl->outputTexture.get() : nullptr;
}

}
