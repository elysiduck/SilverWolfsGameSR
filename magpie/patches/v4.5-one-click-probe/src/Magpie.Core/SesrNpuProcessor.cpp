#include "pch.h"
#include "SesrNpuProcessor.h"

#include "DeviceResources.h"
#include "DirectXHelper.h"
#include "Logger.h"
#include "TextureHelper.h"

#include <array>
#include <atomic>
#include <condition_variable>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <immintrin.h>
#include <mutex>
#include <sstream>
#include <thread>
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

std::vector<uint8_t> ReadBinaryFile(const std::filesystem::path& path) {
	std::ifstream stream(path, std::ios::binary | std::ios::ate);
	if (!stream) {
		throw std::runtime_error("无法读取 GameSR ONNX 文件");
	}
	const std::streamsize size = stream.tellg();
	if (size <= 0) {
		throw std::runtime_error("GameSR ONNX 文件为空");
	}
	stream.seekg(0, std::ios::beg);
	std::vector<uint8_t> bytes((size_t)size);
	if (!stream.read(reinterpret_cast<char*>(bytes.data()), size)) {
		throw std::runtime_error("读取 GameSR ONNX 文件失败");
	}
	return bytes;
}

std::filesystem::path PrepareAsciiVaipRuntime(
	const std::filesystem::path& configPath,
	const std::filesystem::path& xclbinPath) {
	std::wstring localAppData(32768, L'\0');
	const DWORD length = GetEnvironmentVariableW(
		L"LOCALAPPDATA", localAppData.data(), (DWORD)localAppData.size());
	if (!length || length >= localAppData.size()) {
		throw std::runtime_error("无法获取 LOCALAPPDATA 以创建 GameSR NPU 运行目录");
	}
	localAppData.resize(length);
	const std::filesystem::path runtimeDir =
		std::filesystem::path(localAppData) / L"MagpieGameSR" / L"vaip-runtime";
	for (const wchar_t ch : runtimeDir.native()) {
		if (ch > 0x7f) {
			throw std::runtime_error("LOCALAPPDATA 路径不是纯英文，无法创建 VAIP 兼容目录");
		}
	}
	std::error_code ec;
	std::filesystem::create_directories(runtimeDir / L"cache", ec);
	if (ec) {
		throw std::runtime_error("无法创建 GameSR NPU 运行缓存目录");
	}
	std::filesystem::copy_file(configPath, runtimeDir / configPath.filename(),
		std::filesystem::copy_options::overwrite_existing, ec);
	if (ec) {
		throw std::runtime_error("无法复制 VAIP 配置到兼容目录");
	}
	std::filesystem::copy_file(xclbinPath, runtimeDir / xclbinPath.filename(),
		std::filesystem::copy_options::overwrite_existing, ec);
	if (ec) {
		throw std::runtime_error("无法复制 NPU overlay 到兼容目录");
	}
	return runtimeDir;
}

struct SesrModelSelection {
	std::filesystem::path modelPath;
	std::filesystem::path cacheDir;
	std::filesystem::path decoderPath;
	std::string cacheKey;
	bool external = false;
};

std::filesystem::path FindGameSrDecoder(const std::filesystem::path& modelPath) {
	std::vector<std::filesystem::path> candidates;
	candidates.emplace_back(modelPath.wstring() + L".decoder.json");
	std::wstring base = modelPath.filename().wstring();
	const std::wstring marker = L"-int8-qdq.onnx";
	if (base.size() > marker.size() &&
		base.compare(base.size() - marker.size(), marker.size(), marker) == 0) {
		base.resize(base.size() - marker.size());
		candidates.emplace_back(modelPath.parent_path() / (base + L".onnx.decoder.json"));
	}
	for (const auto& candidate : candidates) {
		std::error_code ec;
		if (std::filesystem::is_regular_file(candidate, ec)) return candidate;
	}
	return {};
}


std::filesystem::path FindGameSrFp32Sibling(const std::filesystem::path& modelPath) {
	std::wstring name = modelPath.filename().wstring();
	const std::wstring marker = L"-int8-qdq.onnx";
	if (name.size() <= marker.size() ||
		name.compare(name.size() - marker.size(), marker.size(), marker) != 0) {
		return {};
	}
	name.resize(name.size() - marker.size());
	const std::filesystem::path candidate = modelPath.parent_path() / (name + L".onnx");
	std::error_code ec;
	return std::filesystem::is_regular_file(candidate, ec) ? candidate : std::filesystem::path{};
}

std::wstring GameSrProbeTimestamp() {
	SYSTEMTIME st{};
	GetLocalTime(&st);
	return fmt::format(L"{:04}{:02}{:02}-{:02}{:02}{:02}",
		st.wYear, st.wMonth, st.wDay, st.wHour, st.wMinute, st.wSecond);
}

struct ProbeFloatDiffStats {
	double mae = 0.0;
	double rmse = 0.0;
	double maxAbs = 0.0;
	double cosine = 0.0;
	double rmsA = 0.0;
	double rmsB = 0.0;
	double zeroFractionA = 0.0;
	double zeroFractionB = 0.0;
};

ProbeFloatDiffStats CompareFloatVectors(std::span<const float> a, std::span<const float> b) {
	ProbeFloatDiffStats result{};
	if (a.size() != b.size() || a.empty()) return result;
	double sumAbs = 0.0, sumSq = 0.0, dot = 0.0, normA = 0.0, normB = 0.0;
	uint64_t zeroA = 0, zeroB = 0;
	for (size_t i = 0; i < a.size(); ++i) {
		const double av = a[i], bv = b[i], d = av - bv;
		sumAbs += std::abs(d);
		sumSq += d * d;
		result.maxAbs = std::max(result.maxAbs, std::abs(d));
		dot += av * bv;
		normA += av * av;
		normB += bv * bv;
		if (std::abs(av) < 1e-6) ++zeroA;
		if (std::abs(bv) < 1e-6) ++zeroB;
	}
	const double n = (double)a.size();
	result.mae = sumAbs / n;
	result.rmse = std::sqrt(sumSq / n);
	result.rmsA = std::sqrt(normA / n);
	result.rmsB = std::sqrt(normB / n);
	result.cosine = (normA > 0.0 && normB > 0.0) ? dot / std::sqrt(normA * normB) : 0.0;
	result.zeroFractionA = (double)zeroA / n;
	result.zeroFractionB = (double)zeroB / n;
	return result;
}

struct ProbeImageDiffStats {
	double mae = 0.0;
	double rmse = 0.0;
	double psnr = 0.0;
	double maxAbs = 0.0;
};

ProbeImageDiffStats CompareRgba8(std::span<const uint8_t> a, std::span<const uint8_t> b) {
	ProbeImageDiffStats result{};
	if (a.size() != b.size() || a.size() < 4) return result;
	double sumAbs = 0.0, sumSq = 0.0;
	uint64_t count = 0;
	for (size_t i = 0; i + 3 < a.size(); i += 4) {
		for (size_t c = 0; c < 3; ++c) {
			const double d = ((double)a[i + c] - (double)b[i + c]) / 255.0;
			sumAbs += std::abs(d);
			sumSq += d * d;
			result.maxAbs = std::max(result.maxAbs, std::abs(d));
			++count;
		}
	}
	if (!count) return result;
	result.mae = sumAbs / (double)count;
	const double mse = sumSq / (double)count;
	result.rmse = std::sqrt(mse);
	result.psnr = mse > 0.0 ? 10.0 * std::log10(1.0 / mse) : 99.0;
	return result;
}

bool ExtractJsonNumberArray(std::string_view json, std::string_view key,
	std::vector<float>& values) {
	const std::string quoted = fmt::format("\"{}\"", key);
	size_t pos = json.find(quoted);
	if (pos == std::string_view::npos) return false;
	pos = json.find('[', pos + quoted.size());
	if (pos == std::string_view::npos) return false;
	int depth = 0;
	for (size_t i = pos; i < json.size(); ++i) {
		const char ch = json[i];
		if (ch == '[') { ++depth; continue; }
		if (ch == ']') {
			if (--depth == 0) return true;
			continue;
		}
		if (ch == '-' || ch == '+' || ch == '.' || (ch >= '0' && ch <= '9')) {
			char* end = nullptr;
			const float value = std::strtof(json.data() + i, &end);
			if (end == json.data() + i) return false;
			values.push_back(value);
			i = (size_t)(end - json.data()) - 1;
		}
	}
	return false;
}

bool ExtractJsonString(std::string_view json, std::string_view key, std::string& value) {
	const std::string quoted = fmt::format("\"{}\"", key);
	size_t pos = json.find(quoted);
	if (pos == std::string_view::npos) return false;
	pos = json.find(':', pos + quoted.size());
	if (pos == std::string_view::npos) return false;
	pos = json.find('"', pos + 1);
	if (pos == std::string_view::npos) return false;
	const size_t end = json.find('"', pos + 1);
	if (end == std::string_view::npos) return false;
	value.assign(json.substr(pos + 1, end - pos - 1));
	return true;
}

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
			.decoderPath = {},
			.cacheKey = "modelcachekey_sesr_nhwc_int8_256x256",
			.external = false
		};
	}

	const std::filesystem::path modelsDir = applicationDir / L"models" / L"GameSR";
	const std::filesystem::path cacheDir = applicationDir / L"cache" / L"GameSR";
	std::error_code ec;
	std::filesystem::create_directories(modelsDir, ec);
	std::filesystem::create_directories(cacheDir, ec);

	// Presets store a path relative to models\GameSR. Subdirectories are
	// supported, but absolute paths and traversal outside the model library are
	// rejected. This keeps the old flat layout fully compatible.
	const std::filesystem::path relativeName{ std::wstring(selectedModelName) };
	if (relativeName.empty() || relativeName.is_absolute()) {
		throw std::runtime_error("GameSR 模型路径必须相对于 models\\GameSR");
	}
	const std::filesystem::path selected = std::filesystem::weakly_canonical(modelsDir / relativeName, ec);
	const std::filesystem::path canonicalRoot = std::filesystem::weakly_canonical(modelsDir, ec);
	const auto relativeCheck = selected.lexically_relative(canonicalRoot);
	if (relativeCheck.empty() || *relativeCheck.begin() == L"..") {
		throw std::runtime_error("GameSR 模型路径越出 models\\GameSR");
	}
	if (_wcsicmp(selected.extension().c_str(), L".onnx") != 0 ||
		!std::filesystem::is_regular_file(selected, ec)) {
		throw std::runtime_error(fmt::format(
			"预设绑定的 GameSR 模型不存在: models\\GameSR\\{}",
			relativeName.string()));
	}

	const uint64_t hash = HashFileFNV1a64(selected);
	Logger::Get().Info(fmt::format(
		"GameSR 预设模型: {} (cache {:016x})",
		relativeName.string(), hash));
	return {
		.modelPath = selected,
		.cacheDir = cacheDir,
		.decoderPath = FindGameSrDecoder(selected),
		.cacheKey = fmt::format("modelcachekey_gamesr_contractv3_{:016x}", hash),
		.external = true
	};
}

bool ProbeFullFrameGameSrShape(
	const std::filesystem::path& modelPath,
	uint32_t& width,
	uint32_t& height
) {
	// VitisAI leaves XLNX_ENABLE_CACHE enabled process-wide after the first
	// scaling session.  A later CPU-only metadata probe can then see provider
	// cache state and fail to identify the original FP32/QDQ model contract.
	// Temporarily disable it while inspecting the source ONNX, then restore the
	// caller's environment exactly as it was.
	wchar_t oldCacheValue[32]{};
	const DWORD oldCacheLen = GetEnvironmentVariableW(
		L"XLNX_ENABLE_CACHE", oldCacheValue, (DWORD)std::size(oldCacheValue));
	const bool hadCacheEnv = oldCacheLen > 0 && oldCacheLen < std::size(oldCacheValue);
	SetEnvironmentVariableW(L"XLNX_ENABLE_CACHE", nullptr);
	struct CacheEnvRestore {
		bool hadValue;
		const wchar_t* value;
		~CacheEnvRestore() {
			SetEnvironmentVariableW(L"XLNX_ENABLE_CACHE", hadValue ? value : nullptr);
		}
	} restore{ hadCacheEnv, oldCacheValue };

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
		GameSrLatentNchwF32
	};
	ModelMode modelMode = ModelMode::LegacyTiledInt8;

	// Full-frame GameSR v1.7.1 path. The NPU returns a 720p RGB residual.
	// We upload that residual once and let a compute shader perform both
	// half-pixel bilinear 1.5x sampling and residual composition on the GPU.
	std::vector<float> gameSrInput;
	uint32_t gameSrOutputChannels = 3;
	uint32_t gameSrLatentTextureCount = 1;
	std::vector<float> gameSrLatent;
	std::array<int64_t, 4> gameSrInputShape{ 1, 3, 0, 0 };
	std::array<int64_t, 4> gameSrOutputShape{ 1, 3, 0, 0 };
	std::array<winrt::com_ptr<ID3D11Texture2D>, 2> gameSrLatentTextures;
	std::array<winrt::com_ptr<ID3D11ShaderResourceView>, 2> gameSrLatentSrvs;
	winrt::com_ptr<ID3D11UnorderedAccessView> gameSrOutputUav;
	winrt::com_ptr<ID3D11ComputeShader> gameSrComposeShader;
	winrt::com_ptr<ID3D11ComputeShader> gameSrBaseOnlyShader;
	winrt::com_ptr<ID3D11SamplerState> gameSrLinearSampler;
	winrt::com_ptr<ID3D11Buffer> gameSrDecoderBuffer;
	winrt::com_ptr<ID3D11Buffer> gameSrPhaseDecoderBuffer;
	winrt::com_ptr<ID3D11ShaderResourceView> gameSrPhaseDecoderSrv;
	enum class GameSrDecoderMode : uint32_t {
		Legacy1x1 = 0,
		PhaseAware3x3 = 1
	};
	GameSrDecoderMode gameSrDecoderMode = GameSrDecoderMode::Legacy1x1;
	struct GameSrDecoderConstants {
		float weights[3][8]{};
		float bias[4]{};
		uint32_t decoderMode = 0;
		uint32_t latentChannels = 3;
		uint32_t phaseWeightCount = 0;
		uint32_t reserved = 0;
	} gameSrDecoderConstants;

	enum class AsyncSlotState : uint8_t {
		Free,
		GpuPending,
		CpuQueued,
		WorkerRunning,
		ResultReady
	};

	struct ReadbackSlot {
		winrt::com_ptr<ID3D11Texture2D> texture;
		// Full-frame GameSR additionally keeps a GPU copy of the exact same
		// frame so base and NPU residual never come from different frames.
		winrt::com_ptr<ID3D11Texture2D> gpuTexture;
		winrt::com_ptr<ID3D11ShaderResourceView> gpuSrv;
		winrt::com_ptr<ID3D11Query> readyQuery;
		uint64_t frameId = 0;
		bool pending = false; // legacy tiled path only

		// GameSR v4 async path. D3D resources stay on the render thread; the
		// worker only touches these CPU buffers. A slot is not reused until its
		// result has been consumed, preserving exact base/residual pairing.
		std::atomic<AsyncSlotState> asyncState{ AsyncSlotState::Free };
		std::vector<uint8_t> cpuPixels;
		std::array<std::vector<float>, 2> asyncLatentRgba;
		double workerPackMs = 0.0;
		double workerNpuMs = 0.0;
		double workerOutputPackMs = 0.0;
	};

	static constexpr size_t READBACK_SLOTS = 3;
	std::array<ReadbackSlot, READBACK_SLOTS> readbackSlots;
	size_t nextReadbackSlot = 0;
	uint64_t submittedReadbackFrames = 0;
	uint64_t droppedReadbackFrames = 0;
	uint64_t reusedOutputFrames = 0;

	// GameSR v4: CPU pack + ORT/VitisAI Run live on a dedicated worker.
	// The render thread never calls Ort::Session::Run for the full-frame path.
	std::thread gameSrWorker;
	std::mutex gameSrWorkerMutex;
	std::condition_variable gameSrWorkerCv;
	std::atomic<bool> stopGameSrWorker{ false };
	std::atomic<bool> gameSrWorkerFailed{ false };
	std::atomic<bool> gameSrWorkerBusy{ false };
	std::atomic<uint64_t> workerQueuedFrames{ 0 };
	std::atomic<uint64_t> workerCompletedFrames{ 0 };
	std::atomic<uint64_t> workerDroppedQueuedFrames{ 0 };
	bool gameSrHasOutput = false;


	// GameSR one-click deployment probe. A root-level GameSR-Probe.request file
	// arms exactly one diagnostic run on the next GameSR scaling session.
	bool gameSrProbeRequested = false;
	std::atomic<bool> gameSrProbeQualityDone{ false };
	bool gameSrProbePerfActive = false;
	bool gameSrProbeFinished = false;
	std::filesystem::path gameSrProbeDir;
	std::filesystem::path gameSrProbeModelPath;
	std::filesystem::path gameSrProbeFp32ModelPath;
	std::vector<float> gameSrProbeDecoderWeights;
	std::vector<float> gameSrProbeDecoderBias;
	uint64_t gameSrProbePerfSamples = 0;
	uint64_t gameSrProbeBaseSubmitted = 0;
	uint64_t gameSrProbeBaseDropped = 0;
	uint64_t gameSrProbeBaseWorkerDropped = 0;
	uint64_t gameSrProbeBaseReused = 0;
	double gameSrProbeRenderSum = 0.0;
	double gameSrProbeCopySum = 0.0;
	double gameSrProbePackSum = 0.0;
	double gameSrProbeNpuSum = 0.0;
	double gameSrProbeOutputPackSum = 0.0;
	double gameSrProbeUploadSum = 0.0;
	double gameSrProbeComposeSum = 0.0;
	double gameSrProbeNpuMax = 0.0;
	std::chrono::steady_clock::time_point gameSrProbePerfBegin{};
	static constexpr uint64_t GAME_SR_PROBE_PERF_SAMPLES = 180;

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
		std::wstring_view modelName,
		uint32_t requestedGameSrOutputChannels
	) {
#if !defined(_M_X64)
		(void)resources;
		(void)input;
		(void)requestedGameSrOutputChannels;
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
		if (requestedGameSrOutputChannels && !modelSelection.external) {
			Logger::Get().Error(fmt::format(
				"GameSR NPU O{} 必须在预设中选择 models\\GameSR 下的对应 ONNX 模型",
				requestedGameSrOutputChannels));
			return false;
		}
		const std::filesystem::path& modelPath = modelSelection.modelPath;

		if (modelSelection.external) {
			const std::filesystem::path probeRequest = applicationDir / L"GameSR-Probe.request";
			std::error_code probeEc;
			if (std::filesystem::is_regular_file(probeRequest, probeEc)) {
				gameSrProbeRequested = true;
				gameSrProbeModelPath = modelPath;
				gameSrProbeFp32ModelPath = FindGameSrFp32Sibling(modelPath);
				gameSrProbeDir = applicationDir /
					(L"GameSR-Probe-" + GameSrProbeTimestamp());
				std::filesystem::create_directories(gameSrProbeDir, probeEc);
				std::filesystem::remove(probeRequest, probeEc);
				std::ofstream status(gameSrProbeDir / L"probe.status", std::ios::binary);
				status << "ARMED\nStart GameSR scaling and keep the scene mostly static until probe.done appears.\n";
				Logger::Get().Info(fmt::format(
					"[GAMESR_PROBE] START dir={} qdq={} fp32={}",
					gameSrProbeDir.string(), modelPath.filename().string(),
					gameSrProbeFp32ModelPath.empty() ? "MISSING" : gameSrProbeFp32ModelPath.filename().string()));
			}
		}
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
		std::filesystem::path providerCacheDir = cacheDir;
		std::filesystem::path providerConfigPath = configPath;
		std::filesystem::path providerXclbinPath = xclbinPath;
		if (modelSelection.external) {
			const std::filesystem::path vaipRuntime =
				PrepareAsciiVaipRuntime(configPath, xclbinPath);
			providerCacheDir = vaipRuntime / L"cache";
			providerConfigPath = vaipRuntime / configPath.filename();
			providerXclbinPath = vaipRuntime / xclbinPath.filename();
			Logger::Get().Info(fmt::format(
				"GameSR VAIP 纯英文运行目录: {}", vaipRuntime.string()));
		}

		Logger::Get().Info(fmt::format(
			"SESR ONNX 阶段 1/4: 模型 {}，来源 {}，ORT {}，模式 {}，输入 {}x{}，切片数 {}，目标 {}，缓存 {}",
			modelPath.filename().string(),
			modelSelection.external ? "GameSR 预设" : "内置 SESR",
			OrtGetApiBase()->GetVersionString(),
			fastMode ? "快速（无重叠）" : "画质（16px 重叠）",
			sourceWidth, sourceHeight, tileCount, target, cacheKey));
		Logger::Get().Flush();

		// models\GameSR is a strict GameSR v1.7.x model library.  Do NOT re-probe
		// the source ONNX here: after a VitisAI session has been created, some
		// Ryzen AI runtime builds make later CPU-only ORT metadata probes
		// unreliable for the rest of the process.  The external-library contract
		// is therefore authoritative, while the Session shape below still validates
		// that the selected file really is a 720p NCHW same-size residual model.
		const bool declaredGameSrResidual = modelSelection.external;
		const uint32_t declaredGameSrWidth = modelSelection.external ? 1280u : 0u;
		const uint32_t declaredGameSrHeight = modelSelection.external ? 720u : 0u;

		SetEnvironmentVariableW(L"XLNX_ENABLE_CACHE", L"1");
		ortEnv = std::make_unique<Ort::Env>(
			ORT_LOGGING_LEVEL_VERBOSE, "Magpie-SESR", OrtLogCallback, nullptr);
		Ort::SessionOptions sessionOptions;
		sessionOptions.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
		sessionOptions.SetIntraOpNumThreads(1);
		sessionOptions.SetInterOpNumThreads(1);
		sessionOptions.SetLogSeverityLevel(ORT_LOGGING_LEVEL_VERBOSE);

		const std::unordered_map<std::string, std::string> providerOptions{
			{ "cache_dir", providerCacheDir.string() },
			{ "cache_key", cacheKey },
			{ "config_file", providerConfigPath.string() },
			{ "dd_use_elf_flow", "1" },
			{ "enable_preemption", "1" },
			{ "enable_cache_file_io_in_mem", "0" },
			{ "enable_txn_elf", "1" },
			{ "is_preemptible", "1" },
			{ "target", target },
			{ "xclbin", providerXclbinPath.string() },
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
		// Some Ryzen AI runtime versions convert a file-path model through the
		// active Windows code page. Loading external GameSR models from memory
		// avoids that conversion and supports Unicode install/model directories.
		if (modelSelection.external) {
			const std::vector<uint8_t> modelBytes = ReadBinaryFile(modelPath);
			ortSession = std::make_unique<Ort::Session>(
				*ortEnv, modelBytes.data(), modelBytes.size(), sessionOptions);
		} else {
			ortSession = std::make_unique<Ort::Session>(
				*ortEnv, modelPath.c_str(), sessionOptions);
		}

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
		const bool validOutputChannels = actualOutputShape.size() == 4 &&
			(actualOutputShape[1] == 3 || actualOutputShape[1] == 4 ||
			 actualOutputShape[1] == 6 || actualOutputShape[1] == 8);
		const bool requestedChannelsMatch = requestedGameSrOutputChannels == 0 ||
			(validOutputChannels && actualOutputShape[1] == (int64_t)requestedGameSrOutputChannels);
		const bool isGameSrResidual =
			declaredGameSrResidual &&
			declaredGameSrWidth == sourceWidth &&
			declaredGameSrHeight == sourceHeight &&
			actualInputShape.size() == 4 && actualOutputShape.size() == 4 &&
			actualInputShape[0] == 1 && actualInputShape[1] == 3 &&
			actualOutputShape[0] == 1 && validOutputChannels && requestedChannelsMatch &&
			actualInputShape[2] == (int64_t)sourceHeight &&
			actualInputShape[3] == (int64_t)sourceWidth &&
			actualOutputShape[2] == actualInputShape[2] &&
			actualOutputShape[3] == actualInputShape[3];

		Logger::Get().Info(fmt::format(
			"ONNX 接口诊断: source-contract={}, session input type={}, output type={}, input {}, output {}",
			declaredGameSrResidual ? "GameSR FP32/QDQ residual" : "legacy/unknown",
			(int)inputInfo.GetElementType(), (int)outputInfo.GetElementType(),
			ShapeToString(actualInputShape), ShapeToString(actualOutputShape)));

		if (!isLegacyTiled && !isGameSrResidual) {
			Logger::Get().Error(fmt::format(
				"模型 {} 接口不受支持。支持：① INT8 NHWC [1,256,256,3]->[1,512,512,3] SESR；② GameSR FP32/QDQ NCHW [1,3,H,W]->[1,O,H,W]，O=3/4/6/8。当前 Session 输入 type={} {}，输出 type={} {}。",
				modelPath.filename().string(),
				(int)inputInfo.GetElementType(), ShapeToString(actualInputShape),
				(int)outputInfo.GetElementType(), ShapeToString(actualOutputShape)));
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
			modelMode = ModelMode::GameSrLatentNchwF32;
			// v1.7.1 contract: residual is LR, final image is a 1.5x half-pixel
			// bilinear base plus a 1.5x half-pixel bilinear residual.
			outputWidth = (sourceWidth * 3) / 2;
			outputHeight = (sourceHeight * 3) / 2;
			gameSrOutputChannels = (uint32_t)actualOutputShape[1];
			gameSrLatentTextureCount = (gameSrOutputChannels + 3) / 4;
			gameSrInputShape = { 1, 3, (int64_t)sourceHeight, (int64_t)sourceWidth };
			gameSrOutputShape = { 1, (int64_t)gameSrOutputChannels,
				(int64_t)sourceHeight, (int64_t)sourceWidth };
			const size_t plane = (size_t)sourceWidth * sourceHeight;
			// Pipeline v4 keeps one persistent NCHW input/output pair for the single
			// NPU worker, plus per-slot capture bytes and interleaved residual output.
			// This avoids per-frame allocation without multiplying the large planar
			// FP32 tensors by the number of pipeline slots.
			gameSrInput.resize(plane * 3);
			gameSrLatent.resize(plane * gameSrOutputChannels);
			inputTensors.clear();
			outputTensors.clear();
			inputTensors.emplace_back(Ort::Value::CreateTensor<float>(
				*ortMemoryInfo, gameSrInput.data(), gameSrInput.size(),
				gameSrInputShape.data(), gameSrInputShape.size()));
			outputTensors.emplace_back(Ort::Value::CreateTensor<float>(
				*ortMemoryInfo, gameSrLatent.data(), gameSrLatent.size(),
				gameSrOutputShape.data(), gameSrOutputShape.size()));
			for (ReadbackSlot& slot : readbackSlots) {
				slot.cpuPixels.resize((size_t)sourceWidth * sourceHeight * 4);
				for (size_t latentIndex = 0; latentIndex < slot.asyncLatentRgba.size(); ++latentIndex) {
					if (latentIndex < gameSrLatentTextureCount) {
						slot.asyncLatentRgba[latentIndex].resize(plane * 4);
					} else {
						slot.asyncLatentRgba[latentIndex].clear();
						slot.asyncLatentRgba[latentIndex].shrink_to_fit();
					}
				}
				slot.asyncState.store(AsyncSlotState::Free, std::memory_order_relaxed);
			}

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

			D3D11_TEXTURE2D_DESC latentDesc{};
			latentDesc.Width = sourceWidth;
			latentDesc.Height = sourceHeight;
			latentDesc.MipLevels = 1;
			latentDesc.ArraySize = 1;
			latentDesc.Format = DXGI_FORMAT_R32G32B32A32_FLOAT;
			latentDesc.SampleDesc = { 1, 0 };
			latentDesc.Usage = D3D11_USAGE_DYNAMIC;
			latentDesc.BindFlags = D3D11_BIND_SHADER_RESOURCE;
			latentDesc.CPUAccessFlags = D3D11_CPU_ACCESS_WRITE;
			for (size_t latentIndex = 0; latentIndex < gameSrLatentTextureCount; ++latentIndex) {
				hr = resources.GetD3DDevice()->CreateTexture2D(
					&latentDesc, nullptr, gameSrLatentTextures[latentIndex].put());
				if (FAILED(hr)) {
					Logger::Get().ComError("创建 GameSR latent 上传纹理失败", hr);
					return false;
				}
				hr = resources.GetD3DDevice()->CreateShaderResourceView(
					gameSrLatentTextures[latentIndex].get(), nullptr, gameSrLatentSrvs[latentIndex].put());
				if (FAILED(hr)) {
					Logger::Get().ComError("创建 GameSR latent SRV 失败", hr);
					return false;
				}
				D3D11_MAPPED_SUBRESOURCE zeroMapped{};
				hr = resources.GetD3DDC()->Map(gameSrLatentTextures[latentIndex].get(), 0,
					D3D11_MAP_WRITE_DISCARD, 0, &zeroMapped);
				if (FAILED(hr)) return false;
				for (uint32_t y = 0; y < sourceHeight; ++y) {
					std::memset((uint8_t*)zeroMapped.pData + (size_t)y * zeroMapped.RowPitch,
						0, (size_t)sourceWidth * sizeof(float) * 4);
				}
				resources.GetD3DDC()->Unmap(gameSrLatentTextures[latentIndex].get(), 0);
			}

			gameSrDecoderConstants.latentChannels = gameSrOutputChannels;
			// O3 is the legacy identity residual decoder. O4/O6/O8 accept both
			// the old bilinear+1x1 contract and schema-v3 nine-phase 3x3 residual.
			if (gameSrOutputChannels == 3) {
				gameSrDecoderConstants.weights[0][0] = 1.0f;
				gameSrDecoderConstants.weights[1][1] = 1.0f;
				gameSrDecoderConstants.weights[2][2] = 1.0f;
			} else {
				if (modelSelection.decoderPath.empty()) {
					Logger::Get().Error("O4/O6/O8 模型缺少 .onnx.decoder.json");
					return false;
				}
				std::ifstream decoderStream(modelSelection.decoderPath, std::ios::binary);
				const std::string decoderJson{
					std::istreambuf_iterator<char>(decoderStream),
					std::istreambuf_iterator<char>()
				};
				std::vector<float> weights;
				std::vector<float> bias;
				std::string decoderType;
				std::string decoderOutput;
				if (!ExtractJsonNumberArray(decoderJson, "weight", weights) ||
					!ExtractJsonNumberArray(decoderJson, "bias", bias)) {
					Logger::Get().Error("GameSR decoder.json 无法读取 weight/bias");
					return false;
				}
				ExtractJsonString(decoderJson, "type", decoderType);
				ExtractJsonString(decoderJson, "output", decoderOutput);
				if (gameSrProbeRequested) {
					gameSrProbeDecoderWeights = weights;
					gameSrProbeDecoderBias = bias;
				}
				const size_t oldWeightCount = (size_t)gameSrOutputChannels * 3;
				const size_t phaseWeightCount = (size_t)3 * 3 * 3 * gameSrOutputChannels * 3 * 3;
				if ((decoderType.empty() || decoderType == "bilinear_then_1x1") &&
					weights.size() == oldWeightCount && bias.size() == 3) {
					for (size_t rgb = 0; rgb < 3; ++rgb) {
						for (size_t channel = 0; channel < gameSrOutputChannels; ++channel) {
							gameSrDecoderConstants.weights[rgb][channel] =
								weights[rgb * gameSrOutputChannels + channel];
						}
						gameSrDecoderConstants.bias[rgb] = bias[rgb];
					}
					Logger::Get().Info(fmt::format(
						"GameSR decoder: legacy bilinear_then_1x1，O{}，权重 {}",
						gameSrOutputChannels, weights.size()));
				} else if (decoderType == "phase_aware_3x3_v1" &&
					decoderOutput == "rgb_residual" &&
					weights.size() == phaseWeightCount && bias.size() == 27) {
					gameSrDecoderMode = GameSrDecoderMode::PhaseAware3x3;
					gameSrDecoderConstants.decoderMode = (uint32_t)gameSrDecoderMode;
					// Repack [phase,rgb,channel,ky,kx] into float4 channel groups.
					// The shader then loads each latent float4 once and reuses it for
					// all RGB outputs instead of issuing one texture load per scalar.
					using Float4 = std::array<float, 4>;
					std::vector<Float4> packedPhase;
					const size_t textureCount = gameSrLatentTextureCount;
					const size_t packedWeightCount = (size_t)9 * 3 * 3 * 3 * textureCount;
					packedPhase.reserve(packedWeightCount + 9);
					for (size_t phase = 0; phase < 9; ++phase) {
						for (size_t rgb = 0; rgb < 3; ++rgb) {
							for (size_t ky = 0; ky < 3; ++ky) {
								for (size_t kx = 0; kx < 3; ++kx) {
									for (size_t texture = 0; texture < textureCount; ++texture) {
										Float4 vector{};
										for (size_t component = 0; component < 4; ++component) {
											const size_t channel = texture * 4 + component;
											if (channel < gameSrOutputChannels) {
												const size_t sourceIndex =
													((((phase * 3 + rgb) * gameSrOutputChannels + channel) * 3 + ky) * 3 + kx);
												vector[component] = weights[sourceIndex];
											}
										}
										packedPhase.push_back(vector);
									}
								}
							}
						}
					}
					gameSrDecoderConstants.phaseWeightCount = (uint32_t)packedPhase.size();
					for (size_t phase = 0; phase < 9; ++phase) {
						packedPhase.push_back(Float4{
							bias[phase * 3], bias[phase * 3 + 1], bias[phase * 3 + 2], 0.0f });
					}
					D3D11_BUFFER_DESC phaseDesc{};
					phaseDesc.ByteWidth = (UINT)(packedPhase.size() * sizeof(Float4));
					phaseDesc.Usage = D3D11_USAGE_IMMUTABLE;
					phaseDesc.BindFlags = D3D11_BIND_SHADER_RESOURCE;
					phaseDesc.MiscFlags = D3D11_RESOURCE_MISC_BUFFER_STRUCTURED;
					phaseDesc.StructureByteStride = sizeof(Float4);
					D3D11_SUBRESOURCE_DATA phaseData{ packedPhase.data(), 0, 0 };
					hr = resources.GetD3DDevice()->CreateBuffer(
						&phaseDesc, &phaseData, gameSrPhaseDecoderBuffer.put());
					if (FAILED(hr)) {
						Logger::Get().ComError("创建 GameSR phase 3x3 参数缓冲失败", hr);
						return false;
					}
					D3D11_SHADER_RESOURCE_VIEW_DESC phaseSrvDesc{};
					phaseSrvDesc.Format = DXGI_FORMAT_UNKNOWN;
					phaseSrvDesc.ViewDimension = D3D11_SRV_DIMENSION_BUFFER;
					phaseSrvDesc.Buffer.NumElements = (UINT)packedPhase.size();
					hr = resources.GetD3DDevice()->CreateShaderResourceView(
						gameSrPhaseDecoderBuffer.get(), &phaseSrvDesc, gameSrPhaseDecoderSrv.put());
					if (FAILED(hr)) {
						Logger::Get().ComError("创建 GameSR phase 3x3 参数 SRV 失败", hr);
						return false;
					}
					Logger::Get().Info(fmt::format(
						"GameSR decoder: phase_aware_3x3_v1 rgb_residual，O{}，JSON权重 {}，GPU float4参数 {}+9",
						gameSrOutputChannels, phaseWeightCount, packedWeightCount));
				} else {
					Logger::Get().Error(fmt::format(
						"GameSR decoder 契约/尺寸不匹配: type='{}' output='{}' weight={} bias={}；O{} 需要旧版 {}/3 或 phase3x3 {}/27",
						decoderType, decoderOutput, weights.size(), bias.size(), gameSrOutputChannels,
						oldWeightCount, phaseWeightCount));
					return false;
				}
			}
			D3D11_BUFFER_DESC decoderBufferDesc{};
			decoderBufferDesc.ByteWidth = sizeof(GameSrDecoderConstants);
			decoderBufferDesc.Usage = D3D11_USAGE_IMMUTABLE;
			decoderBufferDesc.BindFlags = D3D11_BIND_CONSTANT_BUFFER;
			D3D11_SUBRESOURCE_DATA decoderData{ &gameSrDecoderConstants, 0, 0 };
			hr = resources.GetD3DDevice()->CreateBuffer(
				&decoderBufferDesc, &decoderData, gameSrDecoderBuffer.put());
			if (FAILED(hr)) {
				Logger::Get().ComError("创建 GameSR decoder 常量失败", hr);
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
Texture2D<float4> Latent0Tex : register(t1);
Texture2D<float4> Latent1Tex : register(t2);
StructuredBuffer<float4> PhaseParams : register(t3);
SamplerState LinearSampler : register(s0);
RWTexture2D<float4> OutputTex : register(u0);
cbuffer Decoder : register(b0) {
    float4 W0a; float4 W0b;
    float4 W1a; float4 W1b;
    float4 W2a; float4 W2b;
    float4 DecoderBias;
    uint DecoderMode;
    uint LatentChannels;
    uint PhaseWeightCount;
    uint Reserved;
};

float4 LatentVectorAt(int2 position, uint textureIndex) {
    uint latentW, latentH;
    Latent0Tex.GetDimensions(latentW, latentH);
    position = clamp(position, int2(0, 0), int2(latentW - 1, latentH - 1));
    return textureIndex == 0
        ? Latent0Tex.Load(int3(position, 0))
        : Latent1Tex.Load(int3(position, 0));
}

float3 PhaseResidual(uint2 outputPosition) {
    uint phaseX = outputPosition.x % 3;
    uint phaseY = outputPosition.y % 3;
    int2 center = int2(
        (outputPosition.x / 3) * 2 + (phaseX == 2 ? 1 : 0),
        (outputPosition.y / 3) * 2 + (phaseY == 2 ? 1 : 0));
    uint phase = phaseY * 3 + phaseX;
    float3 result = PhaseParams[PhaseWeightCount + phase].xyz;
    uint textureCount = (LatentChannels + 3) / 4;
    [loop] for (uint textureIndex = 0; textureIndex < textureCount; ++textureIndex) {
        [unroll] for (uint ky = 0; ky < 3; ++ky) {
            [unroll] for (uint kx = 0; kx < 3; ++kx) {
                float4 latent = LatentVectorAt(
                    center + int2((int)kx - 1, (int)ky - 1), textureIndex);
                uint tapBase = (((phase * 3) * 3 + ky) * 3 + kx) * textureCount + textureIndex;
                uint rgbStride = 3 * 3 * textureCount;
                result.x += dot(latent, PhaseParams[tapBase]);
                result.y += dot(latent, PhaseParams[tapBase + rgbStride]);
                result.z += dot(latent, PhaseParams[tapBase + rgbStride * 2]);
            }
        }
    }
    return result;
}

[numthreads(8, 8, 1)]
void main(uint3 id : SV_DispatchThreadID) {
    uint outW, outH;
    OutputTex.GetDimensions(outW, outH);
    if (id.x >= outW || id.y >= outH) return;
    float2 uv = (float2(id.xy) + 0.5) / float2(outW, outH);
    float3 base = BaseTex.SampleLevel(LinearSampler, uv, 0).rgb;
    if (DecoderMode == 1) {
        float3 residual = PhaseResidual(id.xy);
        OutputTex[id.xy] = float4(saturate(base + residual), 1.0);
        return;
    }
    float4 latent0 = Latent0Tex.SampleLevel(LinearSampler, uv, 0);
    float4 latent1 = Latent1Tex.SampleLevel(LinearSampler, uv, 0);
    float3 residual = float3(
        dot(latent0, W0a) + dot(latent1, W0b) + DecoderBias.x,
        dot(latent0, W1a) + dot(latent1, W1b) + DecoderBias.y,
        dot(latent0, W2a) + dot(latent1, W2b) + DecoderBias.z);
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

			static constexpr std::string_view baseOnlyHlsl = R"(
Texture2D<float4> BaseTex : register(t0);
SamplerState LinearSampler : register(s0);
RWTexture2D<float4> OutputTex : register(u0);
[numthreads(8, 8, 1)]
void main(uint3 id : SV_DispatchThreadID) {
    uint outW, outH;
    OutputTex.GetDimensions(outW, outH);
    if (id.x >= outW || id.y >= outH) return;
    float2 uv = (float2(id.xy) + 0.5) / float2(outW, outH);
    OutputTex[id.xy] = float4(BaseTex.SampleLevel(LinearSampler, uv, 0).rgb, 1.0);
}
)";
			shaderBlob = nullptr;
			if (!DirectXHelper::CompileComputeShader(
				baseOnlyHlsl, "main", shaderBlob.put(), "GameSRBaseOnlyCS")) {
				return false;
			}
			hr = resources.GetD3DDevice()->CreateComputeShader(
				shaderBlob->GetBufferPointer(), shaderBlob->GetBufferSize(), nullptr,
				gameSrBaseOnlyShader.put());
			if (FAILED(hr)) {
				Logger::Get().ComError("创建 GameSR baseline 着色器失败", hr);
				return false;
			}

			if (!StartGameSrWorker()) {
				return false;
			}
			Logger::Get().Info(fmt::format(
				"GameSR ONNX 阶段 4/4: full-frame NPU 会话完成，{}x{} -> O{} latent，GPU learned 1.5x decoder 合成为 {}x{}；异步 worker 已启动",
				sourceWidth, sourceHeight, gameSrOutputChannels, outputWidth, outputHeight));
		}
		Logger::Get().Flush();
		return true;
#endif
	}

	~Impl() {
		StopGameSrWorker();
	}

	void StopGameSrWorker() noexcept {
		stopGameSrWorker.store(true, std::memory_order_release);
		gameSrWorkerCv.notify_all();
		if (gameSrWorker.joinable()) {
			gameSrWorker.join();
		}
	}

	bool StartGameSrWorker() {
		stopGameSrWorker.store(false, std::memory_order_release);
		gameSrWorkerFailed.store(false, std::memory_order_release);
		try {
			gameSrWorker = std::thread([this]() { GameSrWorkerProc(); });
		} catch (const std::exception& e) {
			Logger::Get().Error(fmt::format("启动 GameSR NPU worker 失败: {}", e.what()));
			return false;
		}
		return true;
	}


	bool SaveProbePng(const std::filesystem::path& path, uint32_t width, uint32_t height,
		std::span<uint8_t> rgba) {
		return TextureHelper::SaveTexture(path.c_str(), width, height,
			EffectIntermediateTextureFormat::R8G8B8A8_UNORM, rgba, width * 4);
	}

	void SaveProbeInputPng(const ReadbackSlot& slot) {
		std::vector<uint8_t> rgba((size_t)sourceWidth * sourceHeight * 4);
		const bool bgra = sourceFormat == DXGI_FORMAT_B8G8R8A8_UNORM;
		for (size_t i = 0, pixels = (size_t)sourceWidth * sourceHeight; i < pixels; ++i) {
			const uint8_t* src = slot.cpuPixels.data() + i * 4;
			uint8_t* dst = rgba.data() + i * 4;
			dst[0] = bgra ? src[2] : src[0];
			dst[1] = src[1];
			dst[2] = bgra ? src[0] : src[2];
			dst[3] = 255;
		}
		SaveProbePng(gameSrProbeDir / L"01_input_720p.png", sourceWidth, sourceHeight, rgba);
	}

	float ProbeBilinearChannel(size_t channel, float sourceX, float sourceY) const {
		const int x0raw = (int)std::floor(sourceX);
		const int y0raw = (int)std::floor(sourceY);
		const float tx = sourceX - (float)x0raw;
		const float ty = sourceY - (float)y0raw;
		const int x0 = std::clamp(x0raw, 0, (int)sourceWidth - 1);
		const int y0 = std::clamp(y0raw, 0, (int)sourceHeight - 1);
		const int x1 = std::clamp(x0raw + 1, 0, (int)sourceWidth - 1);
		const int y1 = std::clamp(y0raw + 1, 0, (int)sourceHeight - 1);
		const size_t plane = (size_t)sourceWidth * sourceHeight;
		const float* p = gameSrInput.data() + channel * plane;
		const float a = p[(size_t)y0 * sourceWidth + x0];
		const float b = p[(size_t)y0 * sourceWidth + x1];
		const float c = p[(size_t)y1 * sourceWidth + x0];
		const float d = p[(size_t)y1 * sourceWidth + x1];
		return (a + (b - a) * tx) + ((c + (d - c) * tx) - (a + (b - a) * tx)) * ty;
	}

	float ProbeLatentAt(std::span<const float> latent, size_t channel, int x, int y) const {
		x = std::clamp(x, 0, (int)sourceWidth - 1);
		y = std::clamp(y, 0, (int)sourceHeight - 1);
		const size_t plane = (size_t)sourceWidth * sourceHeight;
		return latent[channel * plane + (size_t)y * sourceWidth + x];
	}

	void RenderProbeCpu(std::span<const float> latent, std::vector<uint8_t>& output,
		std::vector<uint8_t>* residualView = nullptr) const {
		output.resize((size_t)outputWidth * outputHeight * 4);
		if (residualView) residualView->resize(output.size());
		const bool phaseMode = gameSrDecoderMode == GameSrDecoderMode::PhaseAware3x3;
		for (uint32_t oy = 0; oy < outputHeight; ++oy) {
			const float sy = ((float)oy + 0.5f) * (float)sourceHeight / (float)outputHeight - 0.5f;
			for (uint32_t ox = 0; ox < outputWidth; ++ox) {
				const float sx = ((float)ox + 0.5f) * (float)sourceWidth / (float)outputWidth - 0.5f;
				float base[3]{ ProbeBilinearChannel(0, sx, sy), ProbeBilinearChannel(1, sx, sy), ProbeBilinearChannel(2, sx, sy) };
				float residual[3]{};
				if (phaseMode && !gameSrProbeDecoderWeights.empty() && gameSrProbeDecoderBias.size() == 27) {
					const uint32_t phaseX = ox % 3, phaseY = oy % 3;
					const uint32_t phase = phaseY * 3 + phaseX;
					const int centerX = (int)((ox / 3) * 2 + (phaseX == 2 ? 1 : 0));
					const int centerY = (int)((oy / 3) * 2 + (phaseY == 2 ? 1 : 0));
					for (size_t rgb = 0; rgb < 3; ++rgb) {
						float value = gameSrProbeDecoderBias[phase * 3 + rgb];
						for (size_t ch = 0; ch < gameSrOutputChannels; ++ch) {
							for (size_t ky = 0; ky < 3; ++ky) {
								for (size_t kx = 0; kx < 3; ++kx) {
									const size_t wi = ((((phase * 3 + rgb) * gameSrOutputChannels + ch) * 3 + ky) * 3 + kx);
									value += gameSrProbeDecoderWeights[wi] * ProbeLatentAt(
										latent, ch, centerX + (int)kx - 1, centerY + (int)ky - 1);
								}
							}
						}
						residual[rgb] = value;
					}
				} else {
					// Legacy 1x1 path: bilinearly sample each latent channel and apply RGB weights.
					const int x0raw = (int)std::floor(sx), y0raw = (int)std::floor(sy);
					const float tx = sx - x0raw, ty = sy - y0raw;
					for (size_t rgb = 0; rgb < 3; ++rgb) {
						float value = rgb < gameSrProbeDecoderBias.size() ? gameSrProbeDecoderBias[rgb] : gameSrDecoderConstants.bias[rgb];
						for (size_t ch = 0; ch < gameSrOutputChannels; ++ch) {
							auto sample = [&](int x, int y) { return ProbeLatentAt(latent, ch, x, y); };
							const float a = sample(x0raw, y0raw), b = sample(x0raw + 1, y0raw);
							const float c = sample(x0raw, y0raw + 1), d = sample(x0raw + 1, y0raw + 1);
							const float lv = (a + (b - a) * tx) + ((c + (d - c) * tx) - (a + (b - a) * tx)) * ty;
							const float w = (rgb * gameSrOutputChannels + ch < gameSrProbeDecoderWeights.size())
								? gameSrProbeDecoderWeights[rgb * gameSrOutputChannels + ch]
								: gameSrDecoderConstants.weights[rgb][ch];
							value += w * lv;
						}
						residual[rgb] = value;
					}
				}
				const size_t outIndex = ((size_t)oy * outputWidth + ox) * 4;
				for (size_t rgb = 0; rgb < 3; ++rgb) {
					const float value = std::clamp(base[rgb] + residual[rgb], 0.0f, 1.0f);
					output[outIndex + rgb] = (uint8_t)std::lround(value * 255.0f);
					if (residualView) {
						const float vis = std::clamp(0.5f + residual[rgb] * 8.0f, 0.0f, 1.0f);
						(*residualView)[outIndex + rgb] = (uint8_t)std::lround(vis * 255.0f);
					}
				}
				output[outIndex + 3] = 255;
				if (residualView) (*residualView)[outIndex + 3] = 255;
			}
		}
	}

	void RenderProbeBilinear(std::vector<uint8_t>& output) const {
		output.resize((size_t)outputWidth * outputHeight * 4);
		for (uint32_t oy = 0; oy < outputHeight; ++oy) {
			const float sy = ((float)oy + 0.5f) * (float)sourceHeight / (float)outputHeight - 0.5f;
			for (uint32_t ox = 0; ox < outputWidth; ++ox) {
				const float sx = ((float)ox + 0.5f) * (float)sourceWidth / (float)outputWidth - 0.5f;
				const size_t outIndex = ((size_t)oy * outputWidth + ox) * 4;
				for (size_t c = 0; c < 3; ++c) {
					output[outIndex + c] = (uint8_t)std::lround(
						std::clamp(ProbeBilinearChannel(c, sx, sy), 0.0f, 1.0f) * 255.0f);
				}
				output[outIndex + 3] = 255;
			}
		}
	}

	bool RunProbeFp32(std::vector<float>& fp32Latent, double& elapsedMs) {
		if (gameSrProbeFp32ModelPath.empty()) return false;
		wchar_t oldCacheValue[32]{};
		const DWORD oldCacheLen = GetEnvironmentVariableW(
			L"XLNX_ENABLE_CACHE", oldCacheValue, (DWORD)std::size(oldCacheValue));
		const bool hadCache = oldCacheLen > 0 && oldCacheLen < std::size(oldCacheValue);
		SetEnvironmentVariableW(L"XLNX_ENABLE_CACHE", nullptr);
		struct Restore { bool had; const wchar_t* value; ~Restore() { SetEnvironmentVariableW(L"XLNX_ENABLE_CACHE", had ? value : nullptr); } } restore{ hadCache, oldCacheValue };
		Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "Magpie-GameSR-OneClick-Probe");
		Ort::SessionOptions options;
		options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
		options.SetIntraOpNumThreads(2);
		options.SetInterOpNumThreads(1);
		Ort::Session session(env, gameSrProbeFp32ModelPath.c_str(), options);
		if (session.GetInputCount() != 1 || session.GetOutputCount() != 1) return false;
		Ort::AllocatorWithDefaultOptions allocator;
		auto inName = session.GetInputNameAllocated(0, allocator);
		auto outName = session.GetOutputNameAllocated(0, allocator);
		Ort::MemoryInfo memory = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
		Ort::Value inputTensor = Ort::Value::CreateTensor<float>(memory, gameSrInput.data(), gameSrInput.size(),
			gameSrInputShape.data(), gameSrInputShape.size());
		fp32Latent.assign(gameSrLatent.size(), 0.0f);
		Ort::Value outputTensor = Ort::Value::CreateTensor<float>(memory, fp32Latent.data(), fp32Latent.size(),
			gameSrOutputShape.data(), gameSrOutputShape.size());
		const char* inputNames[]{ inName.get() };
		const char* outputNames[]{ outName.get() };
		Ort::RunOptions runOptions;
		const auto begin = std::chrono::steady_clock::now();
		session.Run(runOptions, inputNames, &inputTensor, 1, outputNames, &outputTensor, 1);
		elapsedMs = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - begin).count();
		return true;
	}

	void RunGameSrQualityProbe(ReadbackSlot& slot) {
		if (!gameSrProbeRequested || gameSrProbeQualityDone.load(std::memory_order_acquire)) return;
		try {
			std::ofstream status(gameSrProbeDir / L"probe.status", std::ios::binary | std::ios::trunc);
			status << "QUALITY_RUNNING\nThe first run may pause for several seconds because FP32 CPU inference and CPU decoder reference images are generated.\n";
			SaveProbeInputPng(slot);
			std::vector<float> int8Latent = gameSrLatent;
			std::vector<float> fp32Latent;
			double fp32Ms = 0.0;
			const bool fp32Ok = RunProbeFp32(fp32Latent, fp32Ms);
			std::vector<uint8_t> bilinear, int8Image, int8ResidualView, fp32Image, fp32ResidualView;
			RenderProbeBilinear(bilinear);
			RenderProbeCpu(int8Latent, int8Image, &int8ResidualView);
			SaveProbePng(gameSrProbeDir / L"02_bilinear_1080p.png", outputWidth, outputHeight, bilinear);
			SaveProbePng(gameSrProbeDir / L"04_int8_cpu_decoder_1080p.png", outputWidth, outputHeight, int8Image);
			SaveProbePng(gameSrProbeDir / L"06_int8_residual_view_x8.png", outputWidth, outputHeight, int8ResidualView);
			std::filesystem::copy_file(gameSrProbeDir / L"02_bilinear_1080p.png",
				gameSrProbeDir / L"05_zero_residual_1080p.png",
				std::filesystem::copy_options::overwrite_existing);

			ProbeFloatDiffStats latentStats{};
			ProbeImageDiffStats rgbStats{};
			double fp32ResidualRms = 0.0, int8ResidualRms = 0.0;
			auto residualRms = [](std::span<const uint8_t> image, std::span<const uint8_t> base) {
				double sum = 0.0; uint64_t n = 0;
				for (size_t i = 0; i + 3 < image.size() && i + 3 < base.size(); i += 4) {
					for (size_t c = 0; c < 3; ++c) { const double d = ((double)image[i+c] - base[i+c]) / 255.0; sum += d*d; ++n; }
				}
				return n ? std::sqrt(sum / (double)n) : 0.0;
			};
			int8ResidualRms = residualRms(int8Image, bilinear);
			if (fp32Ok) {
				RenderProbeCpu(fp32Latent, fp32Image, &fp32ResidualView);
				SaveProbePng(gameSrProbeDir / L"03_fp32_cpu_decoder_1080p.png", outputWidth, outputHeight, fp32Image);
				SaveProbePng(gameSrProbeDir / L"07_fp32_residual_view_x8.png", outputWidth, outputHeight, fp32ResidualView);
				latentStats = CompareFloatVectors(fp32Latent, int8Latent);
				rgbStats = CompareRgba8(fp32Image, int8Image);
				fp32ResidualRms = residualRms(fp32Image, bilinear);
			}
			std::ofstream q(gameSrProbeDir / L"quality_metrics.csv", std::ios::binary | std::ios::trunc);
			q << "metric,value\n";
			q << "fp32_available," << (fp32Ok ? 1 : 0) << "\n";
			q << "fp32_cpu_inference_ms," << fp32Ms << "\n";
			q << "latent_mae," << latentStats.mae << "\n";
			q << "latent_rmse," << latentStats.rmse << "\n";
			q << "latent_max_abs," << latentStats.maxAbs << "\n";
			q << "latent_cosine," << latentStats.cosine << "\n";
			q << "latent_fp32_rms," << latentStats.rmsA << "\n";
			q << "latent_int8_rms," << latentStats.rmsB << "\n";
			q << "latent_fp32_zero_fraction," << latentStats.zeroFractionA << "\n";
			q << "latent_int8_zero_fraction," << latentStats.zeroFractionB << "\n";
			q << "rgb_fp32_vs_int8_psnr_db," << rgbStats.psnr << "\n";
			q << "rgb_fp32_vs_int8_mae," << rgbStats.mae << "\n";
			q << "rgb_fp32_vs_int8_rmse," << rgbStats.rmse << "\n";
			q << "rgb_fp32_vs_int8_max_abs," << rgbStats.maxAbs << "\n";
			q << "fp32_residual_rms," << fp32ResidualRms << "\n";
			q << "int8_residual_rms," << int8ResidualRms << "\n";
			q << "residual_rms_ratio_int8_over_fp32," << (fp32ResidualRms > 1e-12 ? int8ResidualRms / fp32ResidualRms : 0.0) << "\n";
			q.flush();
			status.close();
			std::ofstream status2(gameSrProbeDir / L"probe.status", std::ios::binary | std::ios::trunc);
			status2 << "QUALITY_DONE\nCollecting 180 fresh NPU results for performance statistics...\n";
			Logger::Get().Info(fmt::format(
				"[GAMESR_PROBE] QUALITY fp32={} cpu_ms={:.2f} latent_cos={:.6f} rgb_psnr={:.2f} residual_ratio={:.3f}",
				fp32Ok ? 1 : 0, fp32Ms, latentStats.cosine, rgbStats.psnr,
				fp32ResidualRms > 1e-12 ? int8ResidualRms / fp32ResidualRms : 0.0));
			gameSrProbeQualityDone.store(true, std::memory_order_release);
		} catch (const std::exception& e) {
			std::ofstream f(gameSrProbeDir / L"quality_error.txt", std::ios::binary | std::ios::trunc);
			f << e.what() << "\n";
			Logger::Get().Error(fmt::format("[GAMESR_PROBE] quality probe failed: {}", e.what()));
			gameSrProbeQualityDone.store(true, std::memory_order_release);
		}
	}

	void FinishGameSrPerformanceProbe() {
		if (gameSrProbeFinished || !gameSrProbePerfSamples) return;
		const double elapsed = std::chrono::duration<double>(std::chrono::steady_clock::now() - gameSrProbePerfBegin).count();
		const double n = (double)gameSrProbePerfSamples;
		const uint64_t submitted = submittedReadbackFrames - gameSrProbeBaseSubmitted;
		const uint64_t dropped = droppedReadbackFrames - gameSrProbeBaseDropped;
		const uint64_t workerDropped = workerDroppedQueuedFrames.load() - gameSrProbeBaseWorkerDropped;
		const uint64_t reused = reusedOutputFrames - gameSrProbeBaseReused;
		std::ofstream p(gameSrProbeDir / L"performance_metrics.csv", std::ios::binary | std::ios::trunc);
		p << "metric,value\n";
		p << "fresh_results," << gameSrProbePerfSamples << "\n";
		p << "elapsed_seconds," << elapsed << "\n";
		p << "fresh_fps," << (elapsed > 0.0 ? n / elapsed : 0.0) << "\n";
		p << "render_thread_avg_ms," << gameSrProbeRenderSum / n << "\n";
		p << "gpu_to_cpu_copy_avg_ms," << gameSrProbeCopySum / n << "\n";
		p << "input_pack_avg_ms," << gameSrProbePackSum / n << "\n";
		p << "npu_avg_ms," << gameSrProbeNpuSum / n << "\n";
		p << "npu_max_ms," << gameSrProbeNpuMax << "\n";
		p << "output_pack_avg_ms," << gameSrProbeOutputPackSum / n << "\n";
		p << "residual_upload_avg_ms," << gameSrProbeUploadSum / n << "\n";
		p << "gpu_compose_submit_avg_ms," << gameSrProbeComposeSum / n << "\n";
		p << "submitted_delta," << submitted << "\n";
		p << "dropped_capture_result_delta," << dropped << "\n";
		p << "worker_queue_dropped_delta," << workerDropped << "\n";
		p << "reused_output_delta," << reused << "\n";
		p << "reuse_rate_per_fresh," << (n > 0 ? (double)reused / n : 0.0) << "\n";
		p << "drop_rate_per_submitted," << (submitted > 0 ? (double)(dropped + workerDropped) / (double)submitted : 0.0) << "\n";
		p.flush();
		std::ofstream done(gameSrProbeDir / L"probe.done", std::ios::binary | std::ios::trunc);
		done << "DONE\n";
		gameSrProbeFinished = true;
		gameSrProbeRequested = false;
		Logger::Get().Info(fmt::format(
			"[GAMESR_PROBE] DONE fresh_fps={:.2f} npu_avg={:.2f}ms reuse={} drop={} worker_drop={}",
			elapsed > 0.0 ? n / elapsed : 0.0, gameSrProbeNpuSum / n, reused, dropped, workerDropped));
		Logger::Get().Flush();
	}

	void GameSrWorkerProc() noexcept {
		const HRESULT probeComHr = CoInitializeEx(nullptr, COINIT_MULTITHREADED);
		struct ProbeCoUninit {
			HRESULT hr;
			~ProbeCoUninit() { if (SUCCEEDED(hr)) CoUninitialize(); }
		} probeCoUninit{ probeComHr };
		SetThreadDescription(GetCurrentThread(), L"Magpie GameSR NPU Worker");
		SetThreadPriority(GetCurrentThread(), THREAD_PRIORITY_ABOVE_NORMAL);

		try {
			while (!stopGameSrWorker.load(std::memory_order_acquire)) {
				ReadbackSlot* selected = nullptr;
				uint64_t newestFrameId = 0;
				{
					std::unique_lock lock(gameSrWorkerMutex);
					gameSrWorkerCv.wait(lock, [this]() {
						if (stopGameSrWorker.load(std::memory_order_acquire)) return true;
						for (ReadbackSlot& slot : readbackSlots) {
							if (slot.asyncState.load(std::memory_order_acquire) ==
								AsyncSlotState::CpuQueued) return true;
						}
						return false;
					});
					if (stopGameSrWorker.load(std::memory_order_acquire)) break;

					// latest-frame-wins: pick the newest queued CPU frame and discard
					// older queued work before it ever reaches the NPU.
					for (ReadbackSlot& slot : readbackSlots) {
						if (slot.asyncState.load(std::memory_order_acquire) ==
							AsyncSlotState::CpuQueued && slot.frameId >= newestFrameId) {
							selected = &slot;
							newestFrameId = slot.frameId;
						}
					}
					for (ReadbackSlot& slot : readbackSlots) {
						if (&slot == selected) continue;
						if (slot.asyncState.load(std::memory_order_acquire) != AsyncSlotState::CpuQueued) {
							continue;
						}
						if (slot.frameId >= newestFrameId) continue;
						AsyncSlotState expected = AsyncSlotState::CpuQueued;
						if (slot.asyncState.compare_exchange_strong(
							expected, AsyncSlotState::Free,
							std::memory_order_acq_rel, std::memory_order_acquire)) {
							++workerDroppedQueuedFrames;
						}
					}
					if (!selected) continue;
					AsyncSlotState expected = AsyncSlotState::CpuQueued;
					if (!selected->asyncState.compare_exchange_strong(
						expected, AsyncSlotState::WorkerRunning,
						std::memory_order_acq_rel, std::memory_order_acquire)) {
						continue;
					}
				}

				gameSrWorkerBusy.store(true, std::memory_order_release);
				const auto packBegin = std::chrono::steady_clock::now();
				const size_t plane = (size_t)sourceWidth * sourceHeight;
				constexpr float INV_255 = 1.0f / 255.0f;
				const size_t cpuStride = (size_t)sourceWidth * 4;
				const __m128 inv255 = _mm_set1_ps(INV_255);
				const bool bgra = sourceFormat == DXGI_FORMAT_B8G8R8A8_UNORM;
				const __m128i redShuffle = bgra
					? _mm_setr_epi8(2, 6, 10, 14, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1)
					: _mm_setr_epi8(0, 4, 8, 12, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1);
				const __m128i greenShuffle =
					_mm_setr_epi8(1, 5, 9, 13, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1);
				const __m128i blueShuffle = bgra
					? _mm_setr_epi8(0, 4, 8, 12, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1)
					: _mm_setr_epi8(2, 6, 10, 14, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1);
				for (uint32_t y = 0; y < sourceHeight; ++y) {
					const uint8_t* row = selected->cpuPixels.data() + (size_t)y * cpuStride;
					const size_t rowBase = (size_t)y * sourceWidth;
					uint32_t x = 0;
					for (; x + 4 <= sourceWidth; x += 4) {
						const __m128i bgra4 = _mm_loadu_si128(
							reinterpret_cast<const __m128i*>(row + (size_t)x * 4));
						const __m128 red = _mm_mul_ps(_mm_cvtepi32_ps(
							_mm_cvtepu8_epi32(_mm_shuffle_epi8(bgra4, redShuffle))), inv255);
						const __m128 green = _mm_mul_ps(_mm_cvtepi32_ps(
							_mm_cvtepu8_epi32(_mm_shuffle_epi8(bgra4, greenShuffle))), inv255);
						const __m128 blue = _mm_mul_ps(_mm_cvtepi32_ps(
							_mm_cvtepu8_epi32(_mm_shuffle_epi8(bgra4, blueShuffle))), inv255);
						const size_t i = rowBase + x;
						_mm_storeu_ps(gameSrInput.data() + i, red);
						_mm_storeu_ps(gameSrInput.data() + plane + i, green);
						_mm_storeu_ps(gameSrInput.data() + plane * 2 + i, blue);
					}
					for (; x < sourceWidth; ++x) {
						const uint8_t* pixel = row + (size_t)x * 4;
						const size_t i = rowBase + x;
						gameSrInput[i] =
							(bgra ? pixel[2] : pixel[0]) * INV_255;
						gameSrInput[plane + i] = pixel[1] * INV_255;
						gameSrInput[plane * 2 + i] =
							(bgra ? pixel[0] : pixel[2]) * INV_255;
					}
				}
				const auto packEnd = std::chrono::steady_clock::now();

				const char* inputNames[]{ inputName.c_str() };
				const char* outputNames[]{ outputName.c_str() };
				Ort::RunOptions runOptions;
				const auto npuBegin = std::chrono::steady_clock::now();
				ortSession->Run(
					runOptions, inputNames, &inputTensors[0], 1,
					outputNames, &outputTensors[0], 1);
				const auto npuEnd = std::chrono::steady_clock::now();

				if (gameSrProbeRequested && !gameSrProbeQualityDone.load(std::memory_order_acquire)) {
					RunGameSrQualityProbe(*selected);
				}

				// Move NCHW O3/O4/O6/O8 -> the minimum number of RGBA planes.
				// Every group uses the same four-pixel SSE transpose. Missing channels
				// in O3 and the second O6 group are filled with zero vectors once per
				// SIMD block rather than tested for every scalar component.
				const auto outputPackBegin = npuEnd;
				for (size_t packedPlane = 0; packedPlane < gameSrLatentTextureCount; ++packedPlane) {
					float* dst = selected->asyncLatentRgba[packedPlane].data();
					const size_t firstChannel = packedPlane * 4;
					const float* channels[4]{};
					for (size_t component = 0; component < 4; ++component) {
						const size_t channel = firstChannel + component;
						if (channel < gameSrOutputChannels) {
							channels[component] = gameSrLatent.data() + channel * plane;
						}
					}
					size_t i = 0;
					for (; i + 4 <= plane; i += 4) {
						__m128 r0 = channels[0] ? _mm_loadu_ps(channels[0] + i) : _mm_setzero_ps();
						__m128 r1 = channels[1] ? _mm_loadu_ps(channels[1] + i) : _mm_setzero_ps();
						__m128 r2 = channels[2] ? _mm_loadu_ps(channels[2] + i) : _mm_setzero_ps();
						__m128 r3 = channels[3] ? _mm_loadu_ps(channels[3] + i) : _mm_setzero_ps();
						_MM_TRANSPOSE4_PS(r0, r1, r2, r3);
						_mm_storeu_ps(dst + (i + 0) * 4, r0);
						_mm_storeu_ps(dst + (i + 1) * 4, r1);
						_mm_storeu_ps(dst + (i + 2) * 4, r2);
						_mm_storeu_ps(dst + (i + 3) * 4, r3);
					}
					for (; i < plane; ++i) {
						for (size_t component = 0; component < 4; ++component) {
							dst[i * 4 + component] = channels[component] ? channels[component][i] : 0.0f;
						}
					}
				}
				const auto outputPackEnd = std::chrono::steady_clock::now();

				selected->workerPackMs = std::chrono::duration<double, std::milli>(
					packEnd - packBegin).count();
				selected->workerNpuMs = std::chrono::duration<double, std::milli>(
					npuEnd - npuBegin).count();
				selected->workerOutputPackMs = std::chrono::duration<double, std::milli>(
					outputPackEnd - outputPackBegin).count();
				selected->asyncState.store(AsyncSlotState::ResultReady, std::memory_order_release);
				++workerCompletedFrames;
				gameSrWorkerBusy.store(false, std::memory_order_release);
			}
		} catch (const Ort::Exception& e) {
			gameSrWorkerBusy.store(false, std::memory_order_release);
			gameSrWorkerFailed.store(true, std::memory_order_release);
			Logger::Get().Error(fmt::format("GameSR NPU worker ORT 异常: {}", e.what()));
		} catch (const std::exception& e) {
			gameSrWorkerBusy.store(false, std::memory_order_release);
			gameSrWorkerFailed.store(true, std::memory_order_release);
			Logger::Get().Error(fmt::format("GameSR NPU worker 异常: {}", e.what()));
		} catch (...) {
			gameSrWorkerBusy.store(false, std::memory_order_release);
			gameSrWorkerFailed.store(true, std::memory_order_release);
			Logger::Get().Error("GameSR NPU worker 未知异常");
		}
	}

	bool ComposeGameSrResult(ID3D11DeviceContext4* context, ReadbackSlot& slot,
		double& uploadMs, double& composeSubmitMs) {
		const auto uploadBegin = std::chrono::steady_clock::now();
		const size_t rgbaRowBytes = (size_t)sourceWidth * sizeof(float) * 4;
		for (size_t latentIndex = 0; latentIndex < gameSrLatentTextureCount; ++latentIndex) {
			D3D11_MAPPED_SUBRESOURCE mappedLatent{};
			HRESULT hr = context->Map(gameSrLatentTextures[latentIndex].get(), 0,
				D3D11_MAP_WRITE_DISCARD, 0, &mappedLatent);
			if (FAILED(hr)) {
				Logger::Get().ComError("上传 GameSR latent 到 GPU 失败", hr);
				return false;
			}
			for (uint32_t y = 0; y < sourceHeight; ++y) {
				std::memcpy((uint8_t*)mappedLatent.pData + (size_t)y * mappedLatent.RowPitch,
					(const uint8_t*)slot.asyncLatentRgba[latentIndex].data() + (size_t)y * rgbaRowBytes,
					rgbaRowBytes);
			}
			context->Unmap(gameSrLatentTextures[latentIndex].get(), 0);
		}
		const auto uploadEnd = std::chrono::steady_clock::now();

		ID3D11ShaderResourceView* srvs[]{ slot.gpuSrv.get(), gameSrLatentSrvs[0].get(),
			gameSrLatentSrvs[1].get(), gameSrPhaseDecoderSrv.get() };
		ID3D11SamplerState* samplers[]{ gameSrLinearSampler.get() };
		ID3D11UnorderedAccessView* uavs[]{ gameSrOutputUav.get() };
		ID3D11Buffer* decoderBuffers[]{ gameSrDecoderBuffer.get() };
		context->CSSetShader(gameSrComposeShader.get(), nullptr, 0);
		context->CSSetShaderResources(0, 4, srvs);
		context->CSSetSamplers(0, 1, samplers);
		context->CSSetConstantBuffers(0, 1, decoderBuffers);
		context->CSSetUnorderedAccessViews(0, 1, uavs, nullptr);
		context->Dispatch((outputWidth + 7) / 8, (outputHeight + 7) / 8, 1);
		ID3D11UnorderedAccessView* nullUav = nullptr;
		ID3D11ShaderResourceView* nullSrvs[4]{ nullptr, nullptr, nullptr, nullptr };
		context->CSSetUnorderedAccessViews(0, 1, &nullUav, nullptr);
		context->CSSetShaderResources(0, 4, nullSrvs);
		context->CSSetShader(nullptr, nullptr, 0);
		const auto composeEnd = std::chrono::steady_clock::now();
		uploadMs = std::chrono::duration<double, std::milli>(uploadEnd - uploadBegin).count();
		composeSubmitMs = std::chrono::duration<double, std::milli>(composeEnd - uploadEnd).count();
		return true;
	}

	bool ComposeGameSrBaseOnly(ID3D11DeviceContext4* context, ReadbackSlot& slot) {
		ID3D11ShaderResourceView* srvs[]{ slot.gpuSrv.get() };
		ID3D11SamplerState* samplers[]{ gameSrLinearSampler.get() };
		ID3D11UnorderedAccessView* uavs[]{ gameSrOutputUav.get() };
		context->CSSetShader(gameSrBaseOnlyShader.get(), nullptr, 0);
		context->CSSetShaderResources(0, 1, srvs);
		context->CSSetSamplers(0, 1, samplers);
		context->CSSetUnorderedAccessViews(0, 1, uavs, nullptr);
		context->Dispatch((outputWidth + 7) / 8, (outputHeight + 7) / 8, 1);
		ID3D11UnorderedAccessView* nullUav = nullptr;
		ID3D11ShaderResourceView* nullSrvs[1]{ nullptr };
		context->CSSetUnorderedAccessViews(0, 1, &nullUav, nullptr);
		context->CSSetShaderResources(0, 1, nullSrvs);
		context->CSSetShader(nullptr, nullptr, 0);
		return true;
	}

	bool ProcessGameSrAsync(ID3D11Texture2D* input, ID3D11DeviceContext4* context) {
		if (gameSrWorkerFailed.load(std::memory_order_acquire)) {
			Logger::Get().Error("GameSR NPU worker 已失败，停止异步处理");
			return false;
		}

		const auto processBegin = std::chrono::steady_clock::now();
		double readbackCopyMs = 0.0;
		double uploadMs = 0.0;
		double composeSubmitMs = 0.0;
		double workerPackMs = 0.0;
		double workerNpuMs = 0.0;
		double workerOutputPackMs = 0.0;
		bool producedResultThisCall = false;

		// 1) Harvest only the newest completed GPU readback. Older completed
		// captures are stale and are released without entering the CPU/NPU queue.
		ReadbackSlot* newestGpuReady = nullptr;
		uint64_t newestGpuFrameId = 0;
		for (ReadbackSlot& slot : readbackSlots) {
			if (slot.asyncState.load(std::memory_order_acquire) != AsyncSlotState::GpuPending) continue;
			const HRESULT status = context->GetData(
				slot.readyQuery.get(), nullptr, 0, D3D11_ASYNC_GETDATA_DONOTFLUSH);
			if (status == S_OK && slot.frameId >= newestGpuFrameId) {
				newestGpuReady = &slot;
				newestGpuFrameId = slot.frameId;
			}
		}
		if (newestGpuReady) {
			for (ReadbackSlot& slot : readbackSlots) {
				if (&slot == newestGpuReady || slot.frameId >= newestGpuFrameId ||
					slot.asyncState.load(std::memory_order_acquire) != AsyncSlotState::GpuPending) continue;
				const HRESULT status = context->GetData(
					slot.readyQuery.get(), nullptr, 0, D3D11_ASYNC_GETDATA_DONOTFLUSH);
				if (status == S_OK) {
					slot.asyncState.store(AsyncSlotState::Free, std::memory_order_release);
					++droppedReadbackFrames;
				}
			}

			const auto copyBegin = std::chrono::steady_clock::now();
			D3D11_MAPPED_SUBRESOURCE mapped{};
			HRESULT hr = context->Map(newestGpuReady->texture.get(), 0, D3D11_MAP_READ, 0, &mapped);
			if (FAILED(hr)) {
				Logger::Get().ComError("读取 GameSR 异步输入纹理失败", hr);
				return false;
			}
			const size_t rowBytes = (size_t)sourceWidth * 4;
			for (uint32_t y = 0; y < sourceHeight; ++y) {
				std::memcpy(
					newestGpuReady->cpuPixels.data() + (size_t)y * rowBytes,
					(const uint8_t*)mapped.pData + (size_t)y * mapped.RowPitch, rowBytes);
			}
			context->Unmap(newestGpuReady->texture.get(), 0);
			readbackCopyMs = std::chrono::duration<double, std::milli>(
				std::chrono::steady_clock::now() - copyBegin).count();
			{
				std::lock_guard lock(gameSrWorkerMutex);
				newestGpuReady->asyncState.store(AsyncSlotState::CpuQueued, std::memory_order_release);
				++workerQueuedFrames;
			}
			gameSrWorkerCv.notify_one();
		}

		// 2) Consume newest completed NPU result on the render thread. No NPU wait.
		ReadbackSlot* newestResult = nullptr;
		uint64_t newestResultFrameId = 0;
		for (ReadbackSlot& slot : readbackSlots) {
			if (slot.asyncState.load(std::memory_order_acquire) == AsyncSlotState::ResultReady &&
				slot.frameId >= newestResultFrameId) {
				newestResult = &slot;
				newestResultFrameId = slot.frameId;
			}
		}
		if (newestResult) {
			for (ReadbackSlot& slot : readbackSlots) {
				if (&slot != newestResult &&
					slot.asyncState.load(std::memory_order_acquire) == AsyncSlotState::ResultReady &&
					slot.frameId < newestResultFrameId) {
					slot.asyncState.store(AsyncSlotState::Free, std::memory_order_release);
					++droppedReadbackFrames;
				}
			}
			workerPackMs = newestResult->workerPackMs;
			workerNpuMs = newestResult->workerNpuMs;
			workerOutputPackMs = newestResult->workerOutputPackMs;
			if (!ComposeGameSrResult(context, *newestResult, uploadMs, composeSubmitMs)) return false;
			newestResult->asyncState.store(AsyncSlotState::Free, std::memory_order_release);
			gameSrHasOutput = true;
			++frameCount;
			producedResultThisCall = true;
		} else if (gameSrHasOutput) {
			++reusedOutputFrames;
		}

		// 3) Submit current GPU frame into a free slot. This is intentionally
		// after result consumption so a just-freed slot can be reused immediately.
		ReadbackSlot* writeSlot = nullptr;
		for (size_t i = 0; i < READBACK_SLOTS; ++i) {
			ReadbackSlot& candidate = readbackSlots[(nextReadbackSlot + i) % READBACK_SLOTS];
			if (candidate.asyncState.load(std::memory_order_acquire) == AsyncSlotState::Free) {
				writeSlot = &candidate;
				nextReadbackSlot = (nextReadbackSlot + i + 1) % READBACK_SLOTS;
				break;
			}
		}
		if (writeSlot) {
			context->CopyResource(writeSlot->gpuTexture.get(), input);
			context->CopyResource(writeSlot->texture.get(), writeSlot->gpuTexture.get());
			context->End(writeSlot->readyQuery.get());
			writeSlot->frameId = ++submittedReadbackFrames;
			writeSlot->asyncState.store(AsyncSlotState::GpuPending, std::memory_order_release);

			// Avoid an uninitialized/black startup frame. The residual upload
			// texture is initialized to zero, so this is plain bilinear base only.
			if (!gameSrHasOutput && frameCount == 0) {
				ComposeGameSrBaseOnly(context, *writeSlot);
			}
		} else {
			++droppedReadbackFrames;
		}

		if (producedResultThisCall && (frameCount == 1 || frameCount % 60 == 0)) {
			uint32_t gpuPending = 0, cpuQueued = 0, running = 0, results = 0;
			for (ReadbackSlot& slot : readbackSlots) {
				switch (slot.asyncState.load(std::memory_order_acquire)) {
				case AsyncSlotState::GpuPending: ++gpuPending; break;
				case AsyncSlotState::CpuQueued: ++cpuQueued; break;
				case AsyncSlotState::WorkerRunning: ++running; break;
				case AsyncSlotState::ResultReady: ++results; break;
				default: break;
				}
			}
			const double renderThreadMs = std::chrono::duration<double, std::milli>(
				std::chrono::steady_clock::now() - processBegin).count();
			if (gameSrProbeRequested && gameSrProbeQualityDone.load(std::memory_order_acquire) && !gameSrProbeFinished) {
				if (!gameSrProbePerfActive) {
					gameSrProbePerfActive = true;
					gameSrProbePerfBegin = std::chrono::steady_clock::now();
					gameSrProbeBaseSubmitted = submittedReadbackFrames;
					gameSrProbeBaseDropped = droppedReadbackFrames;
					gameSrProbeBaseWorkerDropped = workerDroppedQueuedFrames.load();
					gameSrProbeBaseReused = reusedOutputFrames;
				}
				++gameSrProbePerfSamples;
				gameSrProbeRenderSum += renderThreadMs;
				gameSrProbeCopySum += readbackCopyMs;
				gameSrProbePackSum += workerPackMs;
				gameSrProbeNpuSum += workerNpuMs;
				gameSrProbeOutputPackSum += workerOutputPackMs;
				gameSrProbeUploadSum += uploadMs;
				gameSrProbeComposeSum += composeSubmitMs;
				gameSrProbeNpuMax = std::max(gameSrProbeNpuMax, workerNpuMs);
				if (gameSrProbePerfSamples >= GAME_SR_PROBE_PERF_SAMPLES) {
					FinishGameSrPerformanceProbe();
				}
			}
			Logger::Get().Info(fmt::format(
				"GameSR PIPELINE v4 ASYNC: render线程 {:.2f} ms, GPU->CPU行拷贝 {:.2f} ms, worker input-pack {:.2f} ms, NPU {:.2f} ms, worker output-pack {:.2f} ms, residual上传 {:.2f} ms, GPU合成提交 {:.2f} ms, 处理帧 {}, 提交 {}, worker完成 {}, 捕获/结果丢弃 {}, worker队列丢弃 {}, 复用 {}, slots gpu/cpu/run/result={}/{}/{}/{}, workerBusy={}",
				renderThreadMs, readbackCopyMs, workerPackMs, workerNpuMs, workerOutputPackMs, uploadMs,
				composeSubmitMs, frameCount, submittedReadbackFrames,
				workerCompletedFrames.load(), droppedReadbackFrames,
				workerDroppedQueuedFrames.load(), reusedOutputFrames,
				gpuPending, cpuQueued, running, results,
				gameSrWorkerBusy.load() ? 1 : 0));
		}
		return true;
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
		if (modelMode == ModelMode::GameSrLatentNchwF32) {
			return ProcessGameSrAsync(input, context);
		}
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
			if (modelMode == ModelMode::GameSrLatentNchwF32) {
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

		// Kept only as historical reference; all full-frame GameSR models use
		// ProcessGameSrAsync above.
#if 0
		if (modelMode == ModelMode::GameSrLatentNchwF32) {
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
#endif

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
	std::wstring_view modelName,
	uint32_t gameSrOutputChannels
) noexcept {
	try {
		auto impl = std::make_unique<Impl>();
		if (!impl->Initialize(deviceResources, input, fastMode, modelName, gameSrOutputChannels)) {
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
		name == EXACT_1080P_EFFECT_NAME || name == CURRENT_INPUT_EFFECT_NAME ||
		IsGameSrEffect(name);
}

bool SesrNpuProcessor::IsGameSrEffect(std::string_view name) noexcept {
	return name == GAMESR_O3_EFFECT_NAME || name == GAMESR_O4_EFFECT_NAME ||
		name == GAMESR_O6_EFFECT_NAME || name == GAMESR_O8_EFFECT_NAME;
}

uint32_t SesrNpuProcessor::GetGameSrOutputChannels(std::string_view name) noexcept {
	if (name == GAMESR_O3_EFFECT_NAME) return 3;
	if (name == GAMESR_O4_EFFECT_NAME) return 4;
	if (name == GAMESR_O6_EFFECT_NAME) return 6;
	if (name == GAMESR_O8_EFFECT_NAME) return 8;
	return 0;
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

		// NativePresetModels v2.2 contract: every model placed in models\GameSR
		// is a GameSR v1.7.x 720p NCHW same-size residual core.  This deliberately
		// avoids creating a throw-away ORT Session for metadata probing because
		// VitisAI can poison those probes after the first NPU session in-process.
		// Initialize() performs the definitive Session-shape validation later.
		width = 1280;
		height = 720;
		Logger::Get().Info(fmt::format(
			"GameSR 预设: 模型 {} 按模型库合约请求 Exact 输入 {}x{}",
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
