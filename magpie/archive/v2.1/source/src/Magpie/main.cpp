// Copyright (c) Xu
//
// This program is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.
//
// This program is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU General Public License for more details.
//
// You should have received a copy of the GNU General Public License
// along with this program.  If not, see <https://www.gnu.org/licenses/>.


#include "pch.h"
#include "App.h"
#include "Win32Helper.h"
#include "TouchHelper.h"
#include "CommonSharedConstants.h"
#include "Logger.h"
#include <DbgHelp.h>

using namespace Magpie;
using namespace winrt::Magpie::implementation;

// 将当前目录设为程序所在目录
static void SetWorkingDir() noexcept {
	FAIL_FAST_IF_WIN32_BOOL_FALSE(SetCurrentDirectory(
		Win32Helper::GetExePath().parent_path().c_str()));
}

static void InitializeLogger(const wchar_t* logFilePath) noexcept {
	// 最多两个日志文件，每个最多 500KB
	Logger::Get().Initialize(
		spdlog::level::info,
		logFilePath,
		CommonSharedConstants::MAX_LOG_SIZE,
		1
	);
}

static LONG WINAPI WriteNativeCrashDump(EXCEPTION_POINTERS* exceptionInfo) noexcept {
	const DWORD processId = GetCurrentProcessId();
	SYSTEMTIME now{};
	GetLocalTime(&now);

	wchar_t dumpPath[MAX_PATH]{};
	swprintf_s(
		dumpPath,
		L"logs\\magpie-crash-%04u%02u%02u-%02u%02u%02u-pid%lu.dmp",
		now.wYear, now.wMonth, now.wDay,
		now.wHour, now.wMinute, now.wSecond,
		processId
	);
	CreateDirectoryW(L"logs", nullptr);

	const DWORD exceptionCode = exceptionInfo && exceptionInfo->ExceptionRecord
		? exceptionInfo->ExceptionRecord->ExceptionCode : 0;
	Logger::Get().Critical(fmt::format(
		"原生崩溃: 异常代码 {:#010x}, PID {}。正在写入崩溃转储。",
		exceptionCode, processId));

	wil::unique_hfile dumpFile(CreateFileW(
		dumpPath,
		GENERIC_WRITE,
		FILE_SHARE_READ,
		nullptr,
		CREATE_ALWAYS,
		FILE_ATTRIBUTE_NORMAL,
		nullptr
	));
	if (dumpFile) {
		MINIDUMP_EXCEPTION_INFORMATION dumpException{
			.ThreadId = GetCurrentThreadId(),
			.ExceptionPointers = exceptionInfo,
			.ClientPointers = FALSE
		};
		const MINIDUMP_TYPE dumpType = static_cast<MINIDUMP_TYPE>(
			MiniDumpWithThreadInfo | MiniDumpWithIndirectlyReferencedMemory);
		if (MiniDumpWriteDump(
			GetCurrentProcess(),
			processId,
			dumpFile.get(),
			dumpType,
			exceptionInfo ? &dumpException : nullptr,
			nullptr,
			nullptr
		)) {
			Logger::Get().Critical("崩溃转储已写入 logs 文件夹");
		} else {
			Logger::Get().Win32Critical("写入崩溃转储失败");
		}
	} else {
		Logger::Get().Win32Critical("创建崩溃转储文件失败");
	}
	Logger::Get().Flush();
	return EXCEPTION_EXECUTE_HANDLER;
}

int APIENTRY wWinMain(
	_In_ HINSTANCE /*hInstance*/,
	_In_opt_ HINSTANCE /*hPrevInstance*/,
	_In_ wchar_t* lpCmdLine,
	_In_ int /*nCmdShow*/
) {
#ifdef _DEBUG
	SetThreadDescription(GetCurrentThread(), L"Magpie-主线程");
#endif
	
	// 堆损坏时终止进程
	HeapSetInformation(NULL, HeapEnableTerminationOnCorruption, nullptr, 0);

	SetWorkingDir();

	enum {
		Normal,
		RegisterTouchHelper,
		UnRegisterTouchHelper
	} mode = [&]() {
		if (lpCmdLine == L"-r"sv) {
			return RegisterTouchHelper;
		} else if (lpCmdLine == L"-ur"sv) {
			return UnRegisterTouchHelper;
		} else {
			return Normal;
		}
	}();

	InitializeLogger(mode == Normal ?
		CommonSharedConstants::LOG_PATH :
		CommonSharedConstants::REGISTER_TOUCH_HELPER_LOG_PATH);
	SetUnhandledExceptionFilter(WriteNativeCrashDump);

	Logger::Get().Info(fmt::format("程序启动\n\t版本: {}\n\tOS 版本: {}\n\t管理员: {}\n\tPID: {}",
#ifdef MP_VERSION_STRING
		STRINGIFY(MP_VERSION_STRING),
#elif defined(MP_COMMIT_ID)
		"dev (" STRINGIFY(MP_COMMIT_ID) ")",
#else
		"dev",
#endif
		Win32Helper::GetOSVersion().ToString<char>(),
		Win32Helper::IsProcessElevated() ? "是" : "否",
		GetCurrentProcessId()
	));

	if (mode == RegisterTouchHelper) {
		// 使 TouchHelper 获得 UIAccess 权限
		return Magpie::TouchHelper::Register() ? 0 : 1;
	} else if (mode == UnRegisterTouchHelper) {
		return Magpie::TouchHelper::Unregister() ? 0 : 1;
	}

	winrt::init_apartment(winrt::apartment_type::single_threaded);

	auto& app = App::Get();
	if (!app.Initialize(lpCmdLine)) {
		return 0;
	}

	return app.Run();
}
