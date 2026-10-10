"""原生文件对话框（存档 / 读档）。

Windows 走 ``comdlg32`` 的 ``GetOpenFileNameW`` / ``GetSaveFileNameW``，
不依赖 tkinter（打包 spec 刻意排除了 tk）。其它平台若有 tk 则退回
``tkinter.filedialog``；都没有就返回 ``None``（调用方提示取消）。
"""

from __future__ import annotations

import sys
from pathlib import Path


def ask_open_json(*, initial_dir: str | Path | None = None,
                  title: str = "读取布局") -> Path | None:
    """选一个已有的 ``.json`` 存档；取消返回 ``None``。"""
    return _ask(save=False, initial_dir=initial_dir, initial_file=None,
                title=title)


def ask_save_json(*, initial_dir: str | Path | None = None,
                  initial_file: str | None = None,
                  title: str = "保存布局") -> Path | None:
    """另存为 ``.json``；取消返回 ``None``。"""
    return _ask(save=True, initial_dir=initial_dir, initial_file=initial_file,
                title=title)


def _ask(*, save: bool, initial_dir, initial_file, title: str) -> Path | None:
    initial_dir = str(Path(initial_dir) if initial_dir else Path.cwd())
    initial_file = initial_file or ("layout.json" if save else "")
    if sys.platform == "win32":
        path = _win32_file_dialog(
            save=save, initial_dir=initial_dir, initial_file=initial_file,
            title=title,
        )
    else:
        path = _tk_file_dialog(
            save=save, initial_dir=initial_dir, initial_file=initial_file,
            title=title,
        )
    if not path:
        return None
    result = Path(path)
    if save and result.suffix.lower() != ".json":
        result = result.with_suffix(".json")
    return result


# ---------------------------------------------------------------- Windows

def _win32_file_dialog(*, save: bool, initial_dir: str, initial_file: str,
                       title: str) -> str | None:
    import ctypes
    from ctypes import wintypes

    class OPENFILENAMEW(ctypes.Structure):
        _fields_ = [
            ("lStructSize", wintypes.DWORD),
            ("hwndOwner", wintypes.HWND),
            ("hInstance", wintypes.HINSTANCE),
            ("lpstrFilter", wintypes.LPCWSTR),
            ("lpstrCustomFilter", wintypes.LPWSTR),
            ("nMaxCustFilter", wintypes.DWORD),
            ("nFilterIndex", wintypes.DWORD),
            ("lpstrFile", wintypes.LPWSTR),
            ("nMaxFile", wintypes.DWORD),
            ("lpstrFileTitle", wintypes.LPWSTR),
            ("nMaxFileTitle", wintypes.DWORD),
            ("lpstrInitialDir", wintypes.LPCWSTR),
            ("lpstrTitle", wintypes.LPCWSTR),
            ("Flags", wintypes.DWORD),
            ("nFileOffset", wintypes.WORD),
            ("nFileExtension", wintypes.WORD),
            ("lpstrDefExt", wintypes.LPCWSTR),
            ("lCustData", wintypes.LPARAM),
            ("lpfnHook", wintypes.LPVOID),
            ("lpTemplateName", wintypes.LPCWSTR),
            ("pvReserved", wintypes.LPVOID),
            ("dwReserved", wintypes.DWORD),
            ("FlagsEx", wintypes.DWORD),
        ]

    OFN_FILEMUSTEXIST = 0x00001000
    OFN_PATHMUSTEXIST = 0x00000800
    OFN_OVERWRITEPROMPT = 0x00000002
    OFN_NOCHANGEDIR = 0x00000008
    OFN_HIDEREADONLY = 0x00000004
    OFN_EXPLORER = 0x00080000

    buf = ctypes.create_unicode_buffer(initial_file, 1024)
    filt = "布局存档 (*.json)\0*.json\0所有文件 (*.*)\0*.*\0\0"

    ofn = OPENFILENAMEW()
    ofn.lStructSize = ctypes.sizeof(OPENFILENAMEW)
    ofn.lpstrFilter = filt
    ofn.nFilterIndex = 1
    ofn.lpstrFile = ctypes.cast(buf, wintypes.LPWSTR)
    ofn.nMaxFile = len(buf)
    ofn.lpstrInitialDir = initial_dir
    ofn.lpstrTitle = title
    ofn.lpstrDefExt = "json"
    ofn.Flags = (OFN_EXPLORER | OFN_PATHMUSTEXIST | OFN_HIDEREADONLY
                 | OFN_NOCHANGEDIR)
    if save:
        ofn.Flags |= OFN_OVERWRITEPROMPT
    else:
        ofn.Flags |= OFN_FILEMUSTEXIST

    comdlg = ctypes.windll.comdlg32
    fn = comdlg.GetSaveFileNameW if save else comdlg.GetOpenFileNameW
    fn.argtypes = [ctypes.POINTER(OPENFILENAMEW)]
    fn.restype = wintypes.BOOL
    ok = fn(ctypes.byref(ofn))
    if not ok:
        return None
    return buf.value or None


def _tk_file_dialog(*, save: bool, initial_dir: str, initial_file: str,
                    title: str) -> str | None:
    try:
        import tkinter as tk
        from tkinter import filedialog
    except ImportError:
        return None
    root = tk.Tk()
    root.withdraw()
    try:
        root.attributes("-topmost", True)
    except tk.TclError:
        pass
    kwargs = dict(
        title=title,
        initialdir=initial_dir,
        defaultextension=".json",
        filetypes=[("布局存档", "*.json"), ("所有文件", "*.*")],
    )
    try:
        if save:
            path = filedialog.asksaveasfilename(
                initialfile=initial_file, **kwargs,
            )
        else:
            path = filedialog.askopenfilename(**kwargs)
    finally:
        root.destroy()
    return path or None
